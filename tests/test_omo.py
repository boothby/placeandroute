from unittest import  TestCase
from placeandroute.edgebased.omoplacement import place_edges,match_closest
from chimerautils.chimera import  create
import  networkx as nx

class TestOmoPlacement(TestCase):
    def test_first(self):
        g = nx.complete_bipartite_graph(5,6)
        ret =  place_edges(g)
        print ret
        print "x range", min(ret, key=lambda x:x[1])[1], max(ret, key=lambda x:x[1])[1]
        print "y range", min(ret, key=lambda x:x[2])[2], max(ret, key=lambda x:x[2])[2]

    def test_edgematching(self):
        g = nx.complete_bipartite_graph(5,6)
        ret =  place_edges(g)
        g, _ = create(2,2)
        ret2 = match_closest(ret, g)
        print ret2

