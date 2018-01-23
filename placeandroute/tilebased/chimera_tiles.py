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


def expand_solution(g, chains, cg):
    # type: (nx.Graph, Dict[int, List[int]], nx.Graph) -> Dict[int, List[int]]
    ret = dict()
    alreadymapped = set()
    assert all(cg.has_edge(n1 * 4, n2 * 4) for n1, n2 in g.edges())
    assert all(cg.has_edge(n1 * 4 +1, n2 * 4 +1) for n1, n2 in g.edges())
    assert all(cg.has_edge(n1 * 4 , n2 * 4 +1) for n1, n2, data in g.edges(data=True) if data["capacity"] == 16)


    for problemNode, tile_chain in sorted(chains.items(), key=lambda (k,v): -len(v)):
        # single node case this is easy
        if len(tile_chain) == 1:
            nodegroup = next(iter(tile_chain))
            choice = range(nodegroup * 4, (nodegroup+1)*4)
            choice = filter(lambda  x: x not in alreadymapped, choice)[:1]
            alreadymapped.update(choice)
            ret[problemNode]= choice
            continue

        #long chain. rationale: build the subgraph of possible chains, then trim it
        #first filter already used nodes, then pick a connected component that has at least one choice
        # for each group, trim the choices slowly
        node_choices = dict()
        tile_graph = g.subgraph(tile_chain)
        for tile  in tile_chain:
            node_choices[tile] = [tile*4 + x for x in range(4) if tile*4+x not in alreadymapped]
            assert node_choices[tile]

        has_choice = False

        assert nx.is_connected(tile_graph)

        choices_sg = None # type: nx.Graph
        for ccomp in nx.connected_components(cg.subgraph(reduce(list.__add__, node_choices.itervalues()))):
            if all(any(x in ccomp for x in choiceset) for choiceset in node_choices.itervalues()):
                choices_sg = cg.subgraph(ccomp)
                break

        assert choices_sg is not None
        for tile in tile_chain:
            node_choices[tile] = filter(choices_sg.has_node, node_choices[tile])

        while not nx.is_connected(choices_sg.subgraph(choices[0] for choices in node_choices.itervalues())):
            t1 = random.choice(node_choices.keys())
            for ta, tb in nx.dfs_edges(nx.minimum_spanning_tree(tile_graph), t1):
                find_edges = list((n1, n2) for n1, n2 in product(node_choices[ta], node_choices[tb])
                                  if choices_sg.has_edge(n1, n2))
                if not find_edges:
                    break
                n1, n2 = find_edges[0]
                node_choices[ta].remove(n1)
                node_choices[ta].insert(0, n1)
                node_choices[tb].remove(n2)
                node_choices[tb].insert(0, n2)

        chosen = [x[0] for x in node_choices.values()]

        assert nx.is_connected(cg.subgraph(chosen))
        alreadymapped.update(chosen)
        ret[problemNode] = chosen
    return ret