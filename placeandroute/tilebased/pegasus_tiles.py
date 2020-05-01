import random
from collections import Counter
from functools import reduce
import logging
from itertools import permutations
from typing import Set, Dict

import networkx as nx
from dwave_networkx import pegasus_graph
from six import itervalues, iteritems


def pegasus_tiles(m, _):
    g = pegasus_graph(m, coordinates=True)
    relabeling = dict()
    for a in {0,1}:
        for b in range(m):
            for c in range(m -1):
                for i in range(0, 12, 2):
                    n1, n2 = (a, b, i, c), (a,b,i+1, c)
                    if n1 not in g.nodes:
                        assert n2 not in g.nodes
                        continue
                    g = nx.contracted_nodes(g, n1, n2, self_loops=False)
                    g.node[n1]["capacity"] = 2
                    g.node[n1]["usage"] = 0
                    relabeling[n1] = frozenset([n1,n2])
    nx.relabel_nodes(g, relabeling, copy=False)
    return g, small_choices(g)


def small_choices(g):
    ret = []
    for n1, n2 in g.edges():
        if not next(iter(n1))[0] == next(iter(n2))[0]:
            ret.extend([(n1, n2), (n2, n1)])
    return ret

def big_choices(g):
    ret = []
    for firstnode in g.nodes():
        for restnodes in permutations(g.neighbors(firstnode), 2):
            candidate = (firstnode,) + tuple(restnodes)
            for lastnode in g.neighbors(restnodes[0]):
                if lastnode == firstnode:
                    continue
                candidate += tuple(lastnode)
                if g.subgraph(candidate).number_of_edges() == 4:
                    ret.append(candidate)
    return ret



def load_pegasus(tsize):
    g, chs = pegasus_tiles(tsize, tsize)
    original_graph = pegasus_graph(tsize, coordinates=True)

    logging.info("Hardware stats (compressed): nodes: %d(%d) edges: %d(%d)",
        original_graph.number_of_nodes(),g.number_of_nodes(),
        original_graph.number_of_edges(), g.number_of_edges())

    return chs, g, original_graph


def sample_chain(cg, tile_graph):
    # type: (nx.Graph, nx.Graph) -> Set[int]
    """Sample a qubit chain from a tile chain"""
    chain_choice = set()
    node_choices = lambda tile: iter(tile)
    neighborhood = set()

    for node in nx.dfs_preorder_nodes(tile_graph):
        if not chain_choice:
            node_choice = random.choice([x for x in node_choices(node) if x in cg.nodes])
            #logging.debug("%s", (cg.nodes(), (node)))
        else:
            new_choices = []
            for new_node in node_choices(node):
                if new_node in neighborhood:
                    new_choices.append(new_node)
            assert len(new_choices) > 0, (node, node_choices(node), chain_choice, tile_graph.edges, cg.edges)
            node_choice = random.choice(new_choices)
        chain_choice.add(node_choice)
        neighborhood.update(cg.neighbors(node_choice))

    return chain_choice


def checkEqualIvo(lst):
    return lst.count(lst[0]) == len(lst)


def alternative_chains(cg, tile_graph, count, old, node_to_tile):
    # type: (nx.Graph, nx.Graph, Dict, Set) -> nx.Graph
    """Find alternative qubits in a chain with overused qubits"""

    node_choices = lambda tile: iter(tile)

    # start from the old chain
    expanded = set(old)

    for node in old:
        if count[node] > 1:  # if qubit is overused...
            new_nodes = set(node_to_tile[node])  # type: Set[int]
            expanded.update(new_nodes)
            testsubg = cg.subgraph(expanded)

            # add qubits until they are properly connected
            # (added qubits must have the same degree of the old qubit, otherwise we miss connecting qubits)
            while not all(checkEqualIvo(list(map(testsubg.degree,node_to_tile[x]))) for x in new_nodes):
                newnew = set()
                for n in iter(new_nodes):
                    for x in tile_graph.neighbors(node_to_tile[n]):
                        newnew.update(node_choices(x))
                assert len(newnew.difference(new_nodes)) > 0
                # new_nodes.update(newnew)
                new_nodes = newnew
                expanded.update(newnew)
                testsubg = cg.subgraph(expanded)
                # print expanded

    return cg.subgraph(expanded)


def calc_score(r):
    # type: (Dict) -> (int, Counter)
    """Count the number of overused qubits, return the counter object as well"""
    count = Counter()
    for v in r.values():
        count.update(v)
    return sum(itervalues(count)) - len(count), count


#old method based on annealing, works on pegasus as each compressed node has just 2 qubits
def expand_solution(tile_graph, chains, chimera_graph):
    # type: (nx.Graph, Dict[int, List[int]], nx.Graph) -> Dict[int, List[int]]
    """Expand chains on tiles to chains on qubits. Uses annealing (better ideas are welcome)"""
    ret = dict()
    logging.info("Detailed routing start")
    node_to_tile = dict()
    for tnode in tile_graph.nodes:
        for nnode in tnode:
            node_to_tile[nnode] = tnode

    # todo: sorted shoud be not needed anymore here
    for problemNode, tile_chain in sorted(chains.items(), key=lambda kv: -len(kv[1])):
        tile_subgraph = tile_graph.subgraph(tile_chain)
        assert nx.is_connected(tile_subgraph), (tile_chain,)
        ret[problemNode] = sample_chain(chimera_graph, tile_subgraph)
        best = None
        bestScore = None
        for _ in range(1000):
            score, count = calc_score(ret)

            if best is None or score < bestScore:
                best = ret[problemNode]
                bestScore = score
            if score == 0:
                break
            search_space = alternative_chains(chimera_graph, tile_subgraph, count, ret[problemNode], node_to_tile)
            ret[problemNode] = sample_chain(search_space, tile_subgraph)
        ret[problemNode] = best
        logging.info("Placed constraint %s, score: %d", problemNode, bestScore)
    flatten_assignment(chains, chimera_graph, ret, tile_graph, node_to_tile)

    return ret


def flatten_assignment(chains, chimera_graph, ret, tile_graph, node_to_tile):
    score, count = calc_score(ret)
    stall = 0
    while score > 0:
        # pick a random chain with overused qbits
        k = random.choice([k for k, v in iteritems(ret) if any(count[x] > 1 for x in v)])
        oldv = ret[k]

        problemsubg = tile_graph.subgraph(chains[k])  # chain in tile_space as nx.Graph
        search_space = alternative_chains(chimera_graph, problemsubg, count, oldv, node_to_tile)

        # try hard to find a better chain inbetween the alternatives
        # alternative_chains is expensive, exploit it
        new_score, new_count = 0, None
        for _ in range(10):
            ret[k] = sample_chain((search_space), problemsubg)
            new_score, new_count = calc_score(ret)
            if new_score < score: break

        # todo: verify this condition
        if new_score < score or random.random() < 0.00001 * stall / (new_score - score + 1):
            score = new_score
            count = new_count
            logging.info("Overlapping qubits: %s %s %s", score, "stalled", stall)
            stall = 0
        else:
            ret[k] = oldv
            # slowly rise the temperature
            stall += 1

