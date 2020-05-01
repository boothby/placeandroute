import networkx as nx
from scipy.sparse.linalg import svds
from numpy import dot, sqrt
from six import iteritems

def add_embedding(graph, d = 3):
    # type: (nx.Graph, int) -> None
    if graph.graph.has_key("embedding"):
        return graph.graph["embedding"]
    mat = nx.normalized_laplacian_matrix(graph)
    eigvects, eigvals, _ = svds(mat, d, which="SM", return_singular_vectors="u")
    ret = dict()
    for i, node in enumerate(graph.nodes()):
        ret[node] = eigvects[i, :d]
    graph.graph["has_embedding"] = ret
    return ret

class EmbeddingAstarHeuristic(object):
    def __init__(self, graph, d=3):
        self.graph = graph
        self.embedding = add_embedding(graph, d)

    def prepare_node_map(self, nodemap):
        self.nodemap = nodemap
        self._mapped_emb = dict((nodemap[k], v) for k,v in iteritems(self.embedding))

    def __call__(self, a, b):
        return sqrt(self._mapped_emb[a],self._mapped_emb[b])