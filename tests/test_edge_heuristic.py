from unittest import TestCase
from placeandroute.edgebased.heuristic import *
from chimerautils.chimera import create
import networkx as nx
from os.path import dirname
from placeandroute.problemgraph import parse_cnf,cnf_to_graph

class MinMaxTest(TestCase):
    def testinit(self):
        g,_ = create(4,4)
        terminals = {1: [1, 5, 7], 2 : [16], 3 : [ 15, 18, 20, 21]}
        mm = MinMaxRouter(g, terminals)
        mm.run()
        print mm.get_cost()


class EdgeHeuristicTest(TestCase):
    def test1(self):
        g, _ = create(2,2)
        p = nx.cubical_graph()
        ee = EdgeHeuristic(p,g)
        ee.run()
        print ee

    def test_big_problem(self):
        with open(dirname(__file__) + "/../sgen4-sat-160-8.cnf") as f:
            p = cnf_to_graph(parse_cnf(f))
        g, c = create(16, 16)
        ee = EdgeHeuristic(p, g)
        ee.initial_assignment_bfs(c)
        ee.run()
        print ee.assignment, {k: v.edges() for k,v in ee.routing.result.iteritems()}