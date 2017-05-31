from unittest import TestCase
from edgeplacement import place_edges
from placeandroute.chimera import create
from placeandroute.problemgraph import parse_cnf, cnf_to_graph
#from pysmt.smtlib.parser import get_formula,SmtLib20Parser as SmtLibParser
#from pysmt.shortcuts import get_model,reset_env,Solver
import mathsat
from os.path import dirname
import cStringIO
import networkx as nx
class TestPlace_edges(TestCase):
    def test_first(self):
        #reset_env()
        chim = create(16,16)[0]
        with open(dirname(__file__) + "/../../sgen4-sat-160-8.cnf") as f:
            G = cnf_to_graph(parse_cnf(f))
        #chim = create(2,2)[0]
        #G = nx.complete_graph(5)
        p = place_edges(G, chim)
        #print p
        #script = get_formula(cStringIO.StringIO(str(p)))
        #print script
        #print get_model(script, "msat")
        msatcfg = mathsat.msat_create_config()
        msat = mathsat.msat_create_env(msatcfg)
        mathsat.msat_from_smtlib2(msat, str(p))
        print mathsat.msat_solve(msat)



