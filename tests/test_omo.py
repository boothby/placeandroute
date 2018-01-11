from unittest import  TestCase
from placeandroute.edgebased.omoplacement import *
from placeandroute.edgebased.heuristic import EdgeHeuristic, local_order_edges
from chimerautils.chimera import  create
import  networkx as nx
from os.path import dirname
from placeandroute.problemgraph import parse_cnf,cnf_to_graph

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

    def test_big_problem(self):
        with open(dirname(__file__) + "/../simple60.cnf") as f:
            p = cnf_to_graph(parse_cnf(f))
        g, c = create(16, 16)
        approx = place_edges(p)
        ee = EdgeHeuristic(p, g)
        ee._prefer_center(c)
        ee._prefer_intracell()
        ee.assignment = dict()

        approx = scale_on_arch(approx, g)
        print approx

        choices = ordered_choices(approx, g)

        edge_ordered = local_order_edges(p)


        for edge in edge_ordered:
            clist = []
            if edge not in choices:
                edge = (edge[1], edge[0])
            for a, b in choices[edge]:
                clist.append((a,b))
                clist.append((b,a))
            bchoice = None
            bcost = 0
            for choice in clist:
                ee._insert_edge(edge,choice, effort=9)
                cost = ee._node_cost()
                if bchoice is None or cost < bcost:
                    bchoice = choice
                    bcost = cost
                ee._rip_edge(edge, effort=0)
            ee._insert_edge(edge,bchoice, effort=10)
            #print ee.assignment

        ee.run()
        print ee.assignment, {k: v.edges() for k,v in ee.routing.result.iteritems()}

