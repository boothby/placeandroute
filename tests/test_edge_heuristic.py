from unittest import TestCase
from placeandroute.edgebased.heuristic import *
from chimerautils.chimera import create
import networkx as nx

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