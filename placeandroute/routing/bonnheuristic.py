import random as rand
from typing import Optional, List, Any, Tuple, Dict, FrozenSet, Iterable
import networkx as nx
from math import exp, log
from collections import defaultdict, Counter
from typing import Set


def fast_steiner_tree(graph, voi, heuristic=None):
    # type: (nx.Graph, List[Any]) -> FrozenSet[Any]
    """Finds a low-cost Steiner tree by progressively connecting the terminal vertices"""
    ephnode = "ephemeral"
    voi_iter = iter(voi)
    for node in next(voi_iter):
        graph.add_edge(ephnode, node, weight=0)
    for nset in voi_iter:
        ephdest = "ephdestination"
        for node in nset:
            graph.add_edge(node, ephdest, weight=0)
        path = nx.astar_path(graph, ephnode, ephdest, heuristic=heuristic)
        for nn in path[2:-2]:
            graph.add_edge(ephnode, nn, weight=0)
        for node in nset:
            graph.add_edge(ephnode, node, weight=0)
        graph.remove_node(ephdest)
    treenodes = graph.neighbors(ephnode)
    graph.remove_node(ephnode)
    return frozenset(treenodes)
    #ret =  graph.subgraph(treenodes)
    #return nx.minimum_spanning_tree(ret)


def choice_weighted(l):
    # type: (List[Tuple[Any,float]]) -> Any
    """Choose from a list when each element is paired with a probability"""
    r = rand.random()
    for t, prob in l:
        if r < prob:
            return t
        else:
            r -= prob
    return l[-1]


class MinMaxRouter(object):
    """Router that uses min-max resource allocation"""

    def __init__(self, graph, terminals, epsilon=1):
        # type: (nx.Graph, Dict[Any, Any], Optional[float]) -> None
        self.result = dict()
        self.nodes = terminals.keys()
        self.terminals = terminals
        self.wgraph = nx.DiGraph(graph)
        self.ugraph = graph
        self.epsilon = float(epsilon)
        self._astar_heuristic = defaultdict(lambda: defaultdict(float))

        self._initialize_weights()

    def _heuristic_len(self,a,b):
        return self._astar_heuristic[a][b]

    def _initialize_weights(self):
        """Initializes weights. Sets the edge weights on the terminals, in order to discourage chains passing by terminals"""
        terminals = self.terminals
        for n1, n2 in self.wgraph.edges():
            data = self.wgraph.edges[n1,n2]
            data["weight"] = float(1)
        for _, data in self.wgraph.nodes(data=True):
            data["usage"] = 0

        for vs in terminals.itervalues():
            self.increase_weights(vs)



    def increase_weights(self, nodes):
        # type: (Iterable[Any]) -> None
        """Increase weight on incoming edges on a set of nodes"""
        for node in nodes:
            data = self.wgraph.nodes[node]
            data["usage"] += 1
            usage = data["usage"] / data["capacity"]
            for edge in self.wgraph.in_edges(node):
                data = self.wgraph.edges[edge]
                try:
                    data["weight"] *= exp(self.epsilon * usage)
                except OverflowError:
                    raise
                #self._astar_heuristic[node][edge[1]] = data["weight"]

    def run(self, effort=100):
        # type: (Optional[int]) -> None
        candidatetrees = defaultdict(Counter)
        convex_result = dict()
        #self._astar_heuristic = dict(nx.all_pairs_dijkstra_path_length(self.wgraph))


        if not self.terminals:
            return

        for _ in xrange(effort):
            for node, terminals in self.terminals.iteritems():
                #newtree = make_steiner_tree(self.wgraph, list(terminals))
                #newtree = fast_steiner_tree(self.wgraph, iter(terminals), heuristic=self._heuristic_len)
                term_clusters = nx.connected_components(self.ugraph.subgraph(terminals))
                newtree = fast_steiner_tree(self.wgraph, term_clusters)
                candidatetrees[node][newtree] += 1
                self.increase_weights(newtree)

        ## compact candidatetrees
        for node in self.terminals.keys():
            convex_result[node] = [(t, float(q)/effort) for t,q in candidatetrees[node].most_common()]

        self.derandomize(convex_result, effort)


    def derandomize(self, convex_result, effort):
        # type: (Dict[Any, List[Tuple[FrozenSet[Any], float]]], int) -> None
        ret = dict()
        nodes = self.terminals.keys()
        for node in nodes:
            c = convex_result[node][0][0]
            ret[node] = c

        def invertres(r):
            ret = defaultdict(list)
            for k,vv in r.iteritems():
                for v in vv:
                    ret[v].append(k)
            return ret

        def calcscore(r):
            derandom_coeff = log(2*self.wgraph.number_of_nodes()) # high to discourage overlap
            # use usage - capacity instead of usage/capacity, punish only overusage
            return sum(exp(derandom_coeff*max(0, len(x)-self.wgraph.nodes[n]["capacity"]))
                       for n,x in invertres(r).iteritems())

        score = calcscore(ret)
        for temp in range(effort):
          for node in nodes:
            c = choice_weighted(convex_result[node])
            oldc = ret[node]
            if c == oldc: continue
            ret[node] = c
            newscore = calcscore(ret)
            if newscore < score or rand.random()< 0.1*score/(newscore * (temp+1)):
                score = newscore
            else:
                ret[node] = oldc

        self.result = ret


    def get_nodes(self, node):
        # type: (Any) -> Set[Any]
        return self.result[node]


    def is_present(self, p):
        # type: (Any) -> bool
        return p in self.result