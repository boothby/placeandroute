import networkx as nx

from placeandroute.chimera import create, create2
from placeandroute.display import display_embeddingview
from placeandroute.embed import embed
from placeandroute.problemgraph import cnf_to_graph, parse_cnf

if __name__ == '__main__':
    G = nx.Graph()
    G.add_nodes_from([1,2])
    G.add_edges_from([(1,2)])
    A = nx.Graph()
    A.add_nodes_from([1,2])
    A.add_edge(1,2)
    G, _ = create2(2, 2)
    #with open("sgen4-sat-160-8.cnf") as f:
    with open("fla-350-1.cnf") as f:
        G = cnf_to_graph(parse_cnf(f))
    print G.number_of_nodes()
    A, l = create(16, 16)
    embed(G,A)
    for n, data in G.nodes_iter(True):
        chaingraph = (A.subgraph(data["mapto"]))
        assert nx.is_connected(chaingraph)
    #assert all(A.has_edge(a,b) for _, data in G.nodes_iter(data=True) for a,b in combinations(data["mapto"], 2))
    for vertex in A.nodes_iter():
        if "pred" in A.node[vertex]:
            del A.node[vertex]["pred"]
        A.node[vertex]["mapped"] = str(A.node[vertex]["mapped"])
    print A.node
    display_embeddingview(G, 16)
    #display(A, l)


