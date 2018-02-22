"""Solve the routing problem using max min resource allocation"""
import random as rand
from typing import Optional, List, Any, Tuple, Dict, FrozenSet, Iterable, Set
import networkx as nx
from math import exp, log
from collections import defaultdict, Counter
from typing import Set


def fast_steiner_tree(graph, voi_clusters, heuristic=None):
    # type: (nx.Graph, Iterable[Set[Any]]) -> FrozenSet[Any]
    """Finds a low-cost Steiner tree by progressively connecting the terminal vertices."""
    ephnode = "ephemeral"
    voi_iter = iter(voi_clusters)
    for node in next(voi_iter):
        graph.add_edge(ephnode, node, weight=0)

    for connected_vois in voi_iter:
        ephdest = "ephdestination"
        for node in connected_vois:
            graph.add_edge(node, ephdest, weight=0)

        path = nx.astar_path(graph, ephnode, ephdest, heuristic=heuristic)

        for nn in path[2:-2]:
            graph.add_edge(ephnode, nn, weight=0)
        for node in connected_vois:
            graph.add_edge(ephnode, node, weight=0)
        graph.remove_node(ephdest)

    treenodes = graph.neighbors(ephnode)
    graph.remove_node(ephnode)
    return frozenset(treenodes)


def choice_weighted(l):
    # type: (List[Tuple[Any,float]]) -> Any
    """Choose an element from a list when each element is paired with a probability"""
    assert abs(sum(p for _, p in l) - 1.0) < 1E-5

    r = rand.random()
    for t, prob in l:
        if r < prob:
            return t
        else:
            r -= prob
    return l[-1]


class MinMaxRouter(object):
    """Router that uses min-max resource allocation"""

    def __init__(self, graph, terminals, epsilon=1.0):
        # type: (nx.Graph, Dict[Any, List[Any]], Optional[float]) -> None
        self.result = dict()
        self.terminals = terminals
        self.weights_graph = nx.DiGraph(graph)
        self.original_graph = graph
        self.epsilon = float(epsilon)
        # self._astar_heuristic = defaultdict(lambda: defaultdict(float))

        self._initialize_weights()

    # def _heuristic_len(self,a,b):
    #    """Heuristic distance between nodes"""
    #    return self._astar_heuristic[a][b]

    def _initialize_weights(self):
        """Initializes weights. Sets the edge weights on the terminals, in order to discourage chains passing by
        terminals"""
        for n1, n2 in self.weights_graph.edges():
            data = self.weights_graph.edges[n1, n2]
            data["weight"] = float(1)
        for _, data in self.weights_graph.nodes(data=True):
            data["usage"] = 0

        for vs in self.terminals.itervalues():
            self.increase_weights(vs)

    def increase_weights(self, nodes):
        # type: (Iterable[Any]) -> None
        """Increase weights on all incoming edges on a set of nodes"""
        for node in nodes:
            data = self.weights_graph.nodes[node]
            data["usage"] += 1
            usage = data["usage"] / data["capacity"]
            for edge in self.weights_graph.in_edges(node):
                data = self.weights_graph.edges[edge]
                try:
                    data["weight"] *= exp(self.epsilon * usage)
                except OverflowError:
                    raise  # usage is too high
                # self._astar_heuristic[node][edge[1]] = data["weight"]

    def run(self, effort=100):
        # type: (Optional[int]) -> None
        """Run the fair resource allocation algorithm. For each resource, in round robin fashion, allocate 1/effort of
        the capacity of an approximate Steiner tree. Finally in the randomization step choose one of the candidate
        trees."""

        candidatetrees = defaultdict(Counter)
        convex_result = dict()
        # self._astar_heuristic = dict(nx.all_pairs_dijkstra_path_length(self.wgraph))

        if not self.terminals:
            return

        for _ in xrange(effort):
            for node, terminals in self.terminals.iteritems():
                term_clusters = nx.connected_components(self.original_graph.subgraph(terminals))
                # newtree = fast_steiner_tree(self.wgraph, iter(terminals), heuristic=self._heuristic_len)
                newtree = fast_steiner_tree(self.weights_graph, term_clusters)
                candidatetrees[node][newtree] += 1
                self.increase_weights(newtree)

        # get convex sum of candidate trees
        for node in self.terminals.keys():
            convex_result[node] = [(t, float(q) / effort) for t, q in candidatetrees[node].most_common()]

        self.randomize(convex_result, effort)

    def randomize(self, convex_result, effort):
        # type: (Dict[Any, List[Tuple[FrozenSet[Any], float]]], int) -> None
        """Choose one of the possible trees in order to minimize crossings. Uses simulated annealing"""

        ret = dict()
        resources = self.terminals.keys()

        #start using most probable candidates
        for resource in resources:
            c = convex_result[resource][0][0]
            ret[resource] = c

        #(variable -> qbits) -> (qbit -> variables)
        def invertres(r):
            ret = defaultdict(list)
            for k, vv in r.iteritems():
                for v in vv:
                    ret[v].append(k)
            return ret

        # score(qbit) = coeff ** (overusage)
        def calcscore(r):
            coeff = log(2 * self.weights_graph.number_of_nodes())  # high to discourage overlap
            # use usage - capacity instead of usage/capacity, punish only overusage
            return sum(exp(coeff * max(0, len(x) - self.weights_graph.nodes[n]["capacity"]))
                       for n, x in invertres(r).iteritems())

        score = calcscore(ret)
        for temp in range(effort):
            for resource in resources:
                c = choice_weighted(convex_result[resource])
                oldc = ret[resource]
                if c == oldc:
                    continue
                ret[resource] = c
                newscore = calcscore(ret)
                if newscore < score \
                        or rand.random() < 0.1 * score / (newscore * (temp + 1)):  # todo: test simpler strategies here
                    score = newscore
                else:
                    ret[resource] = oldc

        self.result = ret
