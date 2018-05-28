"""Work with chimera graphs/ tiles"""

import random
from collections import Counter
import logging
import networkx as nx
from six import iteritems, itervalues
from six.moves import range
from typing import List, Dict, Set

from placeandroute.tilebased.detailed_channel_routing import detailed_channel_routing, get_chimera_quotient_graph


def chimeratiles(w, h):
    # type: (int, int) -> (nx.Graph, List[int])
    """Return a compressed version of a chimera graph, where each row in a tile is merged in a single node"""
    ret = nx.Graph()
    i = 0
    grid = []
    choices = []
    for c in range(w):
        row = []
        for r in range(h):
            n1, n2 = 4*i, 4*(i + 1)
            n1 = frozenset(range(n1,n1+4))
            n2 = frozenset(range(n2,n2+4))
            i += 2
            ret.add_nodes_from([n1, n2], capacity=4, usage=0)
            ret.add_edge(n1, n2, capacity=16)
            if row:
                ret.add_edge(n2, row[-1][1], capacity=4)
            if grid:
                ret.add_edge(n1, grid[-1][len(row)][0], capacity=4)
            row.append((n1, n2))
            choices.extend([(n1, n2), (n2, n1)])
        grid.append(row)
    return ret, choices


def sample_chain(cg, tile_graph):
    # type: (nx.Graph, nx.Graph) -> Set[int]
    """Sample a qubit chain from a tile chain"""
    chain_choice = set()
    node_choices = lambda tile: iter(tile)
    neighborhood = set()

    for node in nx.dfs_preorder_nodes(tile_graph):
        if not chain_choice:
            node_choice = random.choice([x for x in node_choices(node) if x in cg.nodes])
            logging.debug("%s", (cg.nodes(), (node)))
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

# http://stackoverflow.com/q/3844948/
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
            while not all(checkEqualIvo(map(testsubg.degree,node_to_tile[x])) for x in new_nodes):
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
    for problemNode, tile_chain in sorted(chains.items(), key=lambda (k, v): -len(v)):
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


def test_chimera_tiles(chimera_graph, tile_graph):
    assert all(chimera_graph.has_edge(n1 * 4, n2 * 4) for n1, n2 in tile_graph.edges)
    assert all(chimera_graph.has_edge(n1 * 4 + 1, n2 * 4 + 1) for n1, n2 in tile_graph.edges)
    assert all(chimera_graph.has_edge(n1 * 4, n2 * 4 + 1) for n1, n2, data in tile_graph.edges.data() if
               data["capacity"] == 16)
    assert all(not chimera_graph.has_edge(n1 * 4, n2 * 4 + 1) for n1, n2, data in tile_graph.edges.data() if
               data["capacity"] != 16)


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

import dwave_networkx as dwnx

def chimeratiles2(w,h):
    hwg = dwnx.chimera_graph(w,h)  # type: nx.Graph
    quotient_graph, quotient_mapping, qubit_mapping = get_chimera_quotient_graph(hwg)
    choices = []
    for a, b, data in quotient_graph.edges(data=True):
        if data["weight"] == 16:
            choices.extend(([a, b], [b, a]))
    for nodeset, data in quotient_graph.nodes(data=True):
        data["capacity"] = len(nodeset)
        data["usage"] = 0
    return quotient_graph, choices


def expand_solution2(_, chains, chimera_graph):
    ret, _ = detailed_channel_routing(chains, chimera_graph)
    return ret