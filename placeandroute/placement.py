import networkx as nx

def do_placement(archGraph, problemGraph):
    l2d = nx.spectral_layout(problemGraph)
    #get arch layout
    #scale

    match_graph = nx.Graph()
    for vert in archGraph.nodes_iter():
        match_graph.add_node(("a", vert))

    for vert in problemGraph.iter():
        match_graph.add_node(("p",  vert))
    #add edges

    match = nx.maximal_matching(match_graph)
    return match

