import random as rand
from collections import defaultdict, Counter
import networkx as nx
from placeandroute.SteinerTree import make_steiner_tree


class MinMaxRouter(object):
    def __init__(self, graph, terminals):
        self.result = dict()
        self.nodes = terminals.keys()
        self.terminals = terminals
        self.wgraph = graph.copy()
        for n1, n2 in self.wgraph.edges_iter():
            self.wgraph.edge[n1][n2]["weight"] = 1

    def increase_weights(self, edges):
        for n1, n2 in edges:
            self.wgraph.edge[n1][n2]["weight"] *= 2000

    def run(self):
        candidatetrees = defaultdict(Counter)
        for _ in xrange(1000):
            for node, terminals in self.terminals.iteritems():
                newtree = make_steiner_tree(self.wgraph, list(terminals))
                candidatetrees[node][newtree] += 1
                self._update_weights(newtree)
        ## compact candidatetrees
        for node in self.terminals.keys():
            ## xxx allow overlapping for now
            self.result[node] = candidatetrees[node].most_common(1)[0][0]

    def _update_weights(self, tree):
        for n1, n2 in tree.edges_iter():
            self.wgraph.edge[n1][n2]["weight"] *= 2000

    def get_tree(self, node):
        return self.result[node]

    def get_cost(self):
        return {n: g.number_of_nodes() for n, g in self.result.iteritems()}


class EdgeHeuristic(object):
    def __init__(self, problemGraph, archGraph):
        self.problemEdges = problemGraph.edges()  # list of pairs
        self.archEdges = archGraph.edges()  # list of pairs
        self.problemGraph = problemGraph  # nx.Graph
        self.archGraph = archGraph  # nx.Graph
        self.rounds = 1000
        self.assignment = None  # dict from problemEdges to archEdges

    def initial_assignment(self):
        if self.assignment is None:
            self.assignment = dict(zip(self.problemEdges, rand.sample(self.archEdges, len(self.problemEdges))))
        self.find_routing()
        return self.assignment

    def choose_ripped_edge(self):
        node_score = self.routing.get_cost()
        ret = None
        best_score = 0
        for k, v in self.assignment.iteritems():
            if ret is None or node_score[k[0]] + node_score[k[1]] > best_score:
                ret = k
                best_score = node_score[k[0]] + node_score[k[1]]
        return ret

    def find_routing(self):
        terminals = defaultdict(set)
        for pe, ae in self.assignment.iteritems():
            terminals[pe[0]].add(ae[0])
            terminals[pe[1]].add(ae[1])

        router = MinMaxRouter(self.archGraph, terminals)
        router.increase_weights(self.assignment.itervalues())
        router.run()

        for n1,n2 in self.archGraph.edges():
            self.archGraph[n1][n2]["weight"] = 1

        self.routing = router

        for tree in router.result.itervalues():
            for n1,n2 in tree.edges_iter():
                self.archGraph[n1][n2]["weight"] *= 2000

    def choose_best_edge(self, probEdge):
        # xxx remember archedge can be flipped! (b, a)
        self._bestEdge = None
        self._bestDist = None
        self._probEdge = probEdge


        for n1, n2 in self.archEdges:
            self._update_optimum(n1, n2)
            self._update_optimum(n2, n1)

        return self._bestEdge

    def _node_distance(self, p, tonode):
        ephemeral = ("ephemeral node")
        nodes = self.routing.get_tree(p).nodes()
        for node in nodes:
            self.archGraph.add_edge(ephemeral, node)
        ret =  nx.shortest_path_length(self.archGraph, tonode, ephemeral)
        self.archGraph.remove_node(ephemeral)
        return ret

    def _update_optimum(self, n1, n2):
        p1, p2 = self._probEdge
        dist = self._node_distance(p1, n1) + self._node_distance(p2, n2)
        if self._bestEdge is None or dist < self._bestDist:
            self._bestEdge = (n1, n2)
            self._bestDist = dist

    def run(self):
        self.initial_assignment()
        for round in xrange(self.rounds):
            choice = self.choose_ripped_edge()
            del self.assignment[choice]
            self.find_routing()
            dest = self.choose_best_edge(choice)
            self.assignment[choice] = dest
            self.find_routing()
            count = Counter()
            print self.routing.get_cost()
            for tree in self.routing.result.values():
                count.update(tree.edges())
            for aed in self.assignment.values():
                count[aed] += 1
            if count.most_common(1)[0][1] == 1:
                break

