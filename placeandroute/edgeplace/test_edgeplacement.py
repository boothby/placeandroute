from unittest import TestCase
from edgeplacement import place_edges
from ..chimera import create
from pysmt.smtlib.parser import get_formula,SmtLib20Parser as SmtLibParser
from pysmt.shortcuts import get_model,reset_env,Solver
import cStringIO
import networkx as nx
class TestPlace_edges(TestCase):
    def test_first(self):
        reset_env()
        chim = create(4,4)[0]
        full = nx.complete_graph(5)
        p = place_edges(full, chim)
        parser = SmtLibParser()
        script = get_formula(cStringIO.StringIO(str(p)))
        print get_model(script, "z3")

