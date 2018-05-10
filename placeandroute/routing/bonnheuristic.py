"""Solve the routing problem using max min resource allocation"""
import random as rand
from collections import defaultdict, Counter
from math import exp, log

import networkx as nx
from networkx import DiGraph
from six import iteritems, itervalues
from six.moves import range
from typing import Optional, List, Any, Tuple, Dict, FrozenSet, Iterable
from typing import Set


def bounded_exp(val, maxval=700.0):
    # exp() throws an exception if the parameter is too high
    return exp(min(maxval, val))


def fast_steiner_tree_old(graph, voi_clusters, heuristic=None):
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


def fast_steiner_tree(graph, voi_clusters, heuristic=None):
    ephstart = "ephstart"
    ephdest = "ephdest"
    voi_iter = iter(voi_clusters)
    graph.add_node(ephdest)
    voi_indexes = dict()
    if heuristic:
        def heur_func(a, b):
            if a not in heuristic:
                return 0
            return min(heuristic[a][x] for xcl in graph.predecessors(b) for x in voi_indexes[xcl])
    else:
        heur_func = None

    for node in next(voi_iter):
        graph.add_edge(ephstart, node, weight=0)

    for voi_index, other_vois in enumerate(voi_iter):
        cluster_node = ("ephcluster", voi_index)
        for node in other_vois:
            graph.add_edge(node, cluster_node, weight=0)
        graph.add_edge(cluster_node, ephdest, weight=0)
        voi_indexes[cluster_node] = other_vois

    while voi_indexes:
        path = nx.astar_path(graph, ephstart, ephdest, heuristic=heur_func)
        for nn in path[2:-3]:
            graph.add_edge(ephstart, nn, weight=0)
        cluster_node = path[-2]
        for node in voi_indexes[cluster_node]:
            graph.add_edge(ephstart, node, weight=0)
        graph.remove_node(cluster_node)
        del voi_indexes[cluster_node]
    graph.remove_node(ephdest)
    treenodes = graph.neighbors(ephstart)
    graph.remove_node(ephstart)
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
    weights_graph = None  # type: DiGraph

    def __init__(self, graph, terminals, epsilon=1.0):
        # type: (nx.Graph, Dict[Any, List[Any]], Optional[float]) -> None
        self.result = dict()
        self.terminals = terminals
        self.weights_graph = nx.DiGraph(graph)
        self.original_graph = graph
        self.epsilon = float(epsilon)
        self.coeff = log(
            sum(d["capacity"] for _, d in self.weights_graph.nodes(data=True)))  # high to discourage overlap
        self._initialize_weights()
        # self._heuristic_dist = dict(nx.all_pairs_dijkstra_path_length(self.weights_graph))
        self._heuristic_dist = None

    def _initialize_weights(self):
        """Initializes weights. Sets the edge weights on the terminals, in order to discourage chains passing by
        terminals"""
        for n1, n2 in self.weights_graph.edges():
            data = self.weights_graph.edges[n1, n2]
            data["weight"] = float(1)
        for _, data in self.weights_graph.nodes(data=True):
            data["usage"] = 0

        for vs in itervalues(self.terminals):
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
                data["weight"] *= bounded_exp(max(0, self.epsilon * usage))
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

        term_clusters = {node: list(nx.connected_components(self.original_graph.subgraph(terminals)))
                         for node, terminals in iteritems(self.terminals)}
        for _ in range(effort):
            for node, terminals in iteritems(self.terminals):
                newtree = fast_steiner_tree(self.weights_graph, term_clusters[node], heuristic=self._heuristic_dist)
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

        # start using most probable candidates
        for resource in resources:
            c = convex_result[resource][0][0]
            ret[resource] = c

        # (variable -> qbits) -> (qbit -> variables)
        def invertres(r):
            ret = defaultdict(list)
            for k, vv in iteritems(r):
                for v in vv:
                    ret[v].append(k)
            return ret

        # score(qbit) = coeff ** (overusage)
        def calcscore(r):
            # use usage - capacity instead of usage/capacity, punish only overusage
            return 1.0 + sum(
                0.0 + bounded_exp(self.coeff * max(0, len(x) - self.weights_graph.nodes[n]["capacity"])) - 1
                for n, x in iteritems(invertres(r)))

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
