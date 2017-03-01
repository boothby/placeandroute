import networkx as nx

from placeandroute.chimera import create, create2
from placeandroute.display import display_embeddingview
from placeandroute.embed import embed

if __name__ == '__main__':
    G = nx.Graph()
    G.add_nodes_from([1,2])
    G.add_edges_from([(1,2)])
    A = nx.Graph()
    A.add_nodes_from([1,2])
    A.add_edge(1,2)
    G, _ = create2(2, 2)
    A, l = create(8, 8)
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
    display_embeddingview(G, 8)
    #display(A, l)


