from unittest import TestCase

from os.path import dirname
from chimerautils.chimera import create
from smtutils.process import Solver

from placeandroute.edgebased.edgeplacement import place_edges
from placeandroute.problemgraph import parse_cnf, cnf_to_graph
class TestPlace_edges(TestCase):
    def test_first(self):
        #reset_env()
        chim = create(16,16)[0]
        with open(dirname(__file__) + "/../sgen4-sat-160-8.cnf") as f:
            G = cnf_to_graph(parse_cnf(f))
        #chim = create(2,2)[0]
        #G = nx.complete_graph(5)
        p = place_edges(G, chim)
        #print p
        #script = get_formula(cStringIO.StringIO(str(p)))
        #print script
        #print get_model(script, "msat")
        solv = Solver("optimathsat")

        print solv.run_formula(str(p))



