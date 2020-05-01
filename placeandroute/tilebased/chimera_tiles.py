"""Work with chimera graphs/ tiles"""
import logging

import networkx as nx
from placeandroute.tilebased.detailed_channel_routing import detailed_channel_routing, get_chimera_quotient_graph


import dwave_networkx as dwnx

def chimera_tiles(w, h=None):
    if h is None: h = w
    hwg = dwnx.chimera_graph(w,n=h)  # type: nx.Graph
    quotient_graph, quotient_mapping, qubit_mapping = get_chimera_quotient_graph(hwg)

    choices = []
    for a, b, data in quotient_graph.edges(data=True):
        if data["weight"] == 16:
            choices.extend(([a, b], [b, a]))
    for nodeset, data in quotient_graph.nodes(data=True):
        data["capacity"] = len(nodeset)
        data["usage"] = 0
    return quotient_graph, choices

def test_chimera_tiles(chimera_graph, tile_graph):
    assert all(chimera_graph.has_edge(n1 * 4, n2 * 4) for n1, n2 in tile_graph.edges)
    assert all(chimera_graph.has_edge(n1 * 4 + 1, n2 * 4 + 1) for n1, n2 in tile_graph.edges)
    assert all(chimera_graph.has_edge(n1 * 4, n2 * 4 + 1) for n1, n2, data in tile_graph.edges.data() if
               data["capacity"] == 16)
    assert all(not chimera_graph.has_edge(n1 * 4, n2 * 4 + 1) for n1, n2, data in tile_graph.edges.data() if
               data["capacity"] != 16)


def load_chimera(tsize):
    g, chs = chimera_tiles(tsize, tsize)
    original_graph = dwnx.chimera_graph(tsize)

    logging.info("Hardware stats (compressed): nodes: %d(%d) edges: %d(%d)",
        original_graph.number_of_nodes(),g.number_of_nodes(),
        original_graph.number_of_edges(), g.number_of_edges())

    return chs, g, original_graph


def expand_solution(_, chains, chimera_graph):
    ret, _ = detailed_channel_routing(chains, chimera_graph)
    return ret
