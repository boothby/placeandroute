import networkx as nx
import typing


def branch_decomp_naive(g):
    # type: (nx.Graph) -> nx.Graph
    ret = list(g.edges)
    g = nx.Graph()
    for e in ret:
        g.add_node(e)
    while len(ret)> 1:
        if len(ret) % 2:
            new = [ret.pop(0)]
        else:
            new = []
        for z in zip(ret[::2],ret[1::2]):
            g.add_node(z)
            g.add_edge(z, z[0])
            g.add_edge(z,z[1])
            new.append(z)
        ret = new
    return g
