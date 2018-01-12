from unittest import TestCase
from placeandroute.tilebased.heuristic import TilePlacementHeuristic, chimeratiles, Constraint
import networkx as nx

class TestTileBased(TestCase):
    def test1(self):
        h = TilePlacementHeuristic([],nx.Graph(), [])
        h.run()


    def test2(self):
        cs = [Constraint([[1,2,3],[4,5]]), Constraint([[2,3,4],[5,6]])]
        g, chs = chimeratiles(3,3)
        h = TilePlacementHeuristic(cs, g, chs)
        h.run()