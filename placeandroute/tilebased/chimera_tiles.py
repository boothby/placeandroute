"""Work with chimera graphs/ tiles"""

import random
import networkx as nx
from typing import List, Dict, Set
from collections import Counter


def chimeratiles(w,h):
    # type: (int, int) -> (nx.Graph, List[int])
    """Return a compressed version of a chimera graph, where each row in a tile is merged in a single node"""
    ret = nx.Graph()
    i = 0
    grid = []
    choices = []
    for c in range(w):
        row = []
        for r in range(h):
            n1, n2 = i, i +1
            i += 2
            ret.add_nodes_from([n1,n2], capacity=4, usage=0)
            ret.add_edge(n1,n2, capacity=16)
            if row:
                ret.add_edge(n2, row[-1][1], capacity=4)
            if grid:
                ret.add_edge(n1,grid[-1][len(row)][0],capacity=4)
            row.append((n1,n2))
            choices.extend([(n1,n2), (n2,n1)])
        grid.append(row)
    return ret, choices


def sample_chain(cg, tile_graph):
    # type: (nx.Graph, nx.Graph) -> Set[int]
    """Sample a qubit chain from a tile chain"""
    chain_choice = set()
    node_choices = lambda tile: range(tile * 4, tile * 4 + 4)
    neighborhood = set()

    for node in nx.dfs_preorder_nodes(tile_graph):
        if not chain_choice:
            node_choice = random.choice([x for x in node_choices(node) if x in cg.nodes])
        else:
            new_choices = []
            for new_node in node_choices(node):
                if new_node in neighborhood:
                    new_choices.append(new_node)
            assert len(new_choices) > 0, (node, node_choices(node), chain_choice, tile_graph.edges(), cg.edges())
            node_choice = random.choice(new_choices)
        chain_choice.add(node_choice)
        neighborhood.update(cg.neighbors(node_choice))

    return chain_choice


def alternative_chains(cg, tile_graph, count, old):
    #type: (nx.Graph, nx.Graph, Dict, Set) -> nx.Graph
    """Find alternative qubits in a chain with overused qubits"""

    node_choices = lambda tile: range(tile * 4, tile * 4 + 4)

    #start from the old chain
    expanded = set(old)

    for node in old:
        if count[node] > 1: # if qubit is overused...
            tile = node//4
            new_nodes = set(node_choices(tile))
            expanded.update(new_nodes)
            testsubg = cg.subgraph(expanded)

            # add qubits until they are properly connected
            # (added qubits must have the same degree of the old qubit, otherwise we miss connecting qubits)
            while any(testsubg.degree(x) != testsubg.degree((x//4)*4) for x in new_nodes) :
                newnew = set()
                for n in iter(new_nodes):
                    for x in tile_graph.neighbors(n // 4):
                        newnew.update(node_choices(x))
                assert len(newnew.difference(new_nodes)) > 0
                #new_nodes.update(newnew)
                new_nodes = newnew
                expanded.update(newnew)
                testsubg = cg.subgraph(expanded)
                #print expanded

    return cg.subgraph(expanded)



def calc_score(r):
    # type: (Dict) -> (int, Counter)
    """Count the number of overused qubits, return the counter object as well"""
    count = Counter()
    for v in r.values():
        count.update(v)
    return sum(count.itervalues()) - len(count), count


def expand_solution(tile_graph, chains, chimera_graph):
    # type: (nx.Graph, Dict[int, List[int]], nx.Graph) -> Dict[int, List[int]]
    """Expand chains on tiles to chains on qubits. Uses annealing (better ideas are welcome)"""
    ret = dict()

    # todo: move these in a unit test for chimera_tiles()
    assert all(chimera_graph.has_edge(n1 * 4, n2 * 4) for n1, n2 in tile_graph.edges())
    assert all(chimera_graph.has_edge(n1 * 4 + 1, n2 * 4 + 1) for n1, n2 in tile_graph.edges())
    assert all(chimera_graph.has_edge(n1 * 4, n2 * 4 + 1) for n1, n2, data in tile_graph.edges(data=True) if data["capacity"] == 16)
    assert all(not chimera_graph.has_edge(n1 * 4, n2 * 4 + 1) for n1, n2, data in tile_graph.edges(data=True) if data["capacity"] != 16)

    # todo: sorted shoud be not needed anymore here
    for problemNode, tile_chain in sorted(chains.items(), key=lambda (k,v): -len(v)):
        tile_subgraph = tile_graph.subgraph(tile_chain)
        assert nx.is_connected(tile_subgraph), (tile_chain,)
        ret[problemNode] = sample_chain(chimera_graph, tile_subgraph)



    score, count = calc_score(ret)
    stall  = 0
    while score > 0:
        # pick a random chain with overused qbits
        k = random.choice([k for k,v in ret.iteritems() if any(count[x] > 1 for x in v)])
        oldv = ret[k]

        problemsubg =tile_graph.subgraph(chains[k]) # chain in tile_space as nx.Graph
        search_space = alternative_chains(chimera_graph, problemsubg, count, oldv)

        # try hard to find a better chain inbetween the alternatives
        # alternative_chains is expensive, exploit it
        for _ in xrange(10):
            ret[k] = sample_chain((search_space), problemsubg)
            new_score, new_count = calc_score(ret)
            if new_score < score: break

        # todo: verify this condition
        if new_score < score or random.random() < 0.000001*stall:
            score = new_score
            count = new_count
            stall = 0
            print "Overlapping qubits when expanding:", score
        else:
            ret[k] = oldv
            # slowly rise the temperature
            stall += 1

    return ret