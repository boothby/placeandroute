import random as rand
from typing import Optional, List, Any, Tuple, Dict, FrozenSet, Iterable
import networkx as nx
from math import exp
from collections import defaultdict, Counter
from typing import Set


def fast_steiner_tree(graph, voi_iter):
    # type: (nx.Graph, List[Any]) -> FrozenSet[Any]
    ephnode = "ephemeral"
    graph.add_edge(ephnode, next(voi_iter), weight=0)
    for n in voi_iter:
        path = nx.astar_path(graph, ephnode, n)
        for nn in path[2:]:
            if nn not in graph.neighbors(ephnode):
                graph.add_edge(ephnode, nn, weight=0)
    treenodes = graph.neighbors(ephnode)
    graph.remove_node(ephnode)
    return frozenset(treenodes)
    #ret =  graph.subgraph(treenodes)
    #return nx.minimum_spanning_tree(ret)


def choice_weighted(l):
    # type: (List[Tuple[Any,float]]) -> Any
    r = rand.random()
    for t, prob in l:
        if r < prob:
            return t
        else:
            r -= prob
    return l[-1]


class MinMaxRouter(object):
    def __init__(self, graph, terminals, epsilon=1):
        # type: (nx.Graph, Dict[Any, Any], Optional[float]) -> None
        self.result = dict()
        self.nodes = terminals.keys()
        self.terminals = terminals
        self.wgraph = nx.DiGraph(graph)
        self.epsilon = epsilon

        self._initialize_weights()


    def _initialize_weights(self):
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


    def run(self, effort):
        # type: (int) -> None
        candidatetrees = defaultdict(Counter)
        if not self.terminals:
            return

        for _ in xrange(effort):
            for node, terminals in self.terminals.iteritems():
                #newtree = make_steiner_tree(self.wgraph, list(terminals))
                newtree = fast_steiner_tree(self.wgraph, iter(terminals))
                candidatetrees[node][newtree] += 1
                self.increase_weights(newtree)

        ## compact candidatetrees
        for node in self.terminals.keys():
            self.result[node] = [(t, float(q)/effort) for t,q in candidatetrees[node].most_common()]
        self.derandomize(effort)


    def derandomize(self,effort):
        # type: (int) -> None
        ret = dict()
        nodes = self.terminals.keys()
        for node in nodes:
            c = self.result[node][0][0]
            ret[node] = c

        def invertres(r):
            ret = defaultdict(list)
            for k,vv in r.iteritems():
                for v in vv:
                    ret[v].append(k)
            return ret

        def calcscore(r):
            return sum(float(5)**len(x) for x in invertres(r).itervalues())

        score = calcscore(ret)
        for _ in range(effort):
          for node in nodes:
            c = choice_weighted(self.result[node])
            oldc = ret[node]
            if c == oldc: continue
            ret[node] = c
            newscore = calcscore(ret)
            if newscore < score:
                score = newscore
            else:
                ret[node] = oldc

        self.result = ret


    def get_nodes(self, node):
        # type: (Any) -> Set[Any]
        return self.result[node]

    def get_cost(self):
        # type: () -> Dict[Any, float]
        return {n: len(g) for n, g in self.result.iteritems()}

    def is_present(self, p):
        # type: (Any) -> bool
        return p in self.result