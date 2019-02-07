from functools import reduce
import logging
import networkx as nx
from dwave_networkx import pegasus_graph


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
    return g, reduce(list.__add__,([(n1,n2), (n2,n1)] for n1,n2 in g.edges() if not next(iter(n1))[:-1] == next(iter(n2))[:-1]))

def load_pegasus(tsize):
    g, chs = pegasus_tiles(tsize, tsize)
    original_graph = pegasus_graph(tsize, coordinates=True)

    logging.info("Hardware stats (compressed): nodes: %d(%d) edges: %d(%d)",
        original_graph.number_of_nodes(),g.number_of_nodes(),
        original_graph.number_of_edges(), g.number_of_edges())

    return chs, g, original_graph