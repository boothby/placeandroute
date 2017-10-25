import random as rand
from collections import defaultdict

class MinMaxRouter(object):
    def __init__(self, graph, terminals):
        pass

    def run(self):
        pass

    def get_tree(self, node):
        pass

    def get_cost(self):
        pass # for each problem node, how much space does it take

class EdgeHeuristic(object):

    def __init__(self, problemGraph, archGraph):
        self.problemEdges = problemGraph.edges() # list of pairs
        self.archEdges = archGraph.edges() # list of pairs
        self.problemGraph = problemGraph # nx.Graph
        self.archGraph = archGraph # nx.Graph
        self.rounds = 1000
        self.assignment = None # dict from problemEdges to archEdges


    def initial_assignment(self):
        if self.assignment is None:
            return NotImplementedError # XXX implement random assinment
        self.find_routing()
        return self.assignment

    def choose_ripped_edge(self):
        node_score = self.routing.get_cost()
        ret = None
        best_score = 0
        for k,v in self.assignment:
            if ret is None or node_score[k[0]] + node_score[k[1]] > best_score:
                ret = v
                best_score = node_score[k[0]] + node_score[k[1]]
        return ret

    def find_routing(self):
        terminals = defaultdict(list)
        for pe, ae in self.assignment:
            terminals[pe[0]].append(ae[0])
            terminals[pe[1]].append(ae[1])

        router = MinMaxRouter(self.archGraph, terminals)
        router.run()

        self.routing = router

    def choose_best_edge(self, probEdge):
        #xxx remember archedge can be flipped! (b, a)
        for aEdge in self.archEdges:
            distance = None # distance from nodes in probEdge



    def run(self):
        self.initial_assignment()
        for round in xrange(self.rounds):
            choice = self.choose_ripped_edge()
            del self.assignment[choice]
            self.find_routing()
            dest = self.choose_best_edge(choice)
            self.assignment[choice] = dest


