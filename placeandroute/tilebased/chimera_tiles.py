import random

import networkx as nx
from typing import List, Dict
from itertools import product


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
                ret.add_edge(n1, row[-1][0], capacity=4)
            if grid:
                ret.add_edge(n2,grid[-1][len(row)][1],capacity=4)
            row.append((n1,n2))
            choices.extend([(n1,n2), (n2,n1)])
        grid.append(row)
    return ret, choices


def sample_chain(cg, tile_graph):
    chain_choice = None
    node_choices = lambda tile: range(tile * 4, tile * 4 + 4)

    for node in nx.dfs_preorder_nodes(tile_graph):
        if not chain_choice:
            chain_choice = {random.choice([x for x in node_choices(node) if x in cg.nodes])}
        else:
            new_choices = []
            for new_node in node_choices(node):
                if any(cg.has_edge(old_node, new_node) for old_node in chain_choice):
                    new_choices.append(chain_choice.union({new_node}))
            assert len(new_choices) > 0, (node, node_choices(node), chain_choice, tile_graph.edges(), cg.edges())
            chain_choice = random.choice(new_choices)
    return chain_choice


def improve_chain(cg, tile_graph, count, old):
    node_choices = lambda tile: range(tile * 4, tile * 4 + 4)

    expanded = set(old)
    #if len(old) == 1:
    #    expanded.update(node_choices(next(iter(old))//4))
    #    return cg.subgraph(expanded), tile_graph)

    for node in old:
        if count[node] > 1:
            tile = node//4
            new_nodes = set(node_choices(tile))
            expanded.update(new_nodes)
            testsubg = cg.subgraph(expanded)
            while any(testsubg.degree(x) != testsubg.degree((x//4)*4) for x in new_nodes) :
                newnew = set()
                for n in iter(new_nodes):
                    for x in tile_graph.neighbors(n // 4):
                        newnew.update(node_choices(x))
                assert len(newnew.difference(new_nodes)) > 0
                new_nodes.update(newnew)
                expanded.update(newnew)
                testsubg = cg.subgraph(expanded)
                #print expanded

    return cg.subgraph(expanded)



from collections import Counter
def calc_score(r):
    count = Counter()
    for v in r.values():
        count.update(v)
    return sum(count.itervalues()) - len(count), count


def expand_solution(g, chains, cg):
    # type: (nx.Graph, Dict[int, List[int]], nx.Graph) -> Dict[int, List[int]]
    ret = dict()

    assert all(cg.has_edge(n1 * 4, n2 * 4) for n1, n2 in g.edges())
    assert all(cg.has_edge(n1 * 4 +1, n2 * 4 +1) for n1, n2 in g.edges())
    assert all(cg.has_edge(n1 * 4 , n2 * 4 +1) for n1, n2, data in g.edges(data=True) if data["capacity"] == 16)

    for problemNode, tile_chain in sorted(chains.items(), key=lambda (k,v): -len(v)):
        tile_graph = g.subgraph(tile_chain)
        assert nx.is_connected(tile_graph)
        ret[problemNode] = sample_chain(cg, tile_graph)



    score, count = calc_score(ret)
    while score > 0:
        k = random.choice([k for k,v in ret.iteritems() if any(count[x] > 1 for x in v)])
        oldv = ret[k]
        #for v in oldv:
        #    count[v] -=1
        #del ret[k]
        problemsubg =g.subgraph(chains[k])
        search_space = improve_chain(cg, problemsubg, count, oldv)
        for _ in xrange(10):
            ret[k] = sample_chain((search_space), problemsubg)
            new_score, new_count = calc_score(ret)
            if new_score < score: break
        if new_score <= score or random.random() < 0.001:
            score = new_score
            count = new_count
            print "Overlapping qubits when expanding:", score
        else:
            ret[k] = oldv
    #print ("final score:", score)

    return ret