import random
from unittest import TestCase
from collections import defaultdict

class PlaceAndRoute(object):
    def __init__(self, topology):
        self.embedded_nodes = []
        self.best_embedding = []
        self.best_cost = 1E250
        self.embedding = None
        self.topology = topology

    def closest_node(self, neighbors):
        scores = defaultdict(int)
        retpaths = defaultdict(dict)
        for node in [self.embedded_nodes[x] for x in neighbors]:
            closest = self.topology.single_source_shortest_path(node)
            for k,v in closest.iteritems():
                scores[k] += len(v)
                retpaths[k][node] = v
        ret = min(scores)
        return ret, retpaths[ret]

    def add_route(self, node, to,  paths):
        for path in paths:
            for edge in path:
                self.topology.edges[edge].weight *= 2
        else:
            center = self.topology.vertices[len(self.topology.vertices) // 2]
        self.embedded_nodes.append(node)

    def rip_up(self, node):
        mynode = self.embedded_nodes[node]
        chains = mynode.chains
        for path in chains:
            for edge in path:
                self.topology.edges[edge].weight /= 2
        self.embedded_nodes.remove(mynode)

    def node_cost(self, node):
        mynode = node
        return 1 + sum(x.weight for x in mynode.chains)

    def total_cost(self):
        return sum(x.weight for x in self.topology.edges)

    def run(self, source):
        for node in source:
            to, paths = self.closest_node(node)
            self.add_route(node, to, paths)
        self.improve_effort(1000)

    def improve_step(self):
        for node in self.embedded_nodes:
            oldcost = self.node_cost(node)
            self.rip_up(node)
            newroute = self.closest_node(node.neigbors)
            self.add_route(node, newroute)
            newcost = self.node_cost(node)
            if newcost < oldcost:
                return True
        return False

    def check_best(self):
        newcost = self.total_cost()
        if newcost < self.best_cost:
            self.best_cost = newcost
            self.best_embedding = self.embedding

    def get_best(self):
        return self.best_embedding

    def improve_effort(self, effort):
        for i in xrange(effort):
            if not self.improve_step():
                self.check_best()
                random.shuffle(self.embedded_nodes)
            else:
                self.embedded_nodes.sort(key=self.node_cost)
        self.check_best()

class PlaceAndRoute(TestCase):
    def test_usage(self):
        h = chimera
        pr = PlaceAndRoute(h)
        pr.run(source)
        print pr.get_best()
        pr.improve_effort(100)
        print pr.get_best()