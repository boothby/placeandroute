import logging

from tests.utils import parse_2in4

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(asctime)s %(processName)s %(message)s')

from itertools import product
from multiprocessing import Pool
from os.path import dirname
from unittest import TestCase

import dwave_networkx as dwnx
import networkx as nx
from six import iteritems, print_

from placeandroute.problemgraph import parse_cnf
from placeandroute.tilebased.chimera_tiles import load_chimera, expand_solution
from placeandroute.tilebased.heuristic import TilePlacementHeuristic, Constraint
from placeandroute.tilebased.parallel import ParallelPlacementHeuristic
from placeandroute.tilebased.utils import cnf_to_constraints, show_result

def mapl(func, seq):
    return list(map(func, seq))

def test_result(tile_graph, cnf, heur):
    chains = heur.chains
    for constraint in cnf:
        assert constraint in heur.constraint_placement, heur.constraint_placement
        var_rows = constraint.tile
        for v1, v2 in product(*var_rows):
            if v1 == v2: continue
            ch1 = chains[v1]
            ch2 = chains[v2]
            assert nx.is_connected(tile_graph.subgraph(ch1))
            assert nx.is_connected(tile_graph.subgraph(ch2))
            assert any(tile_graph.has_edge(n1, n2) for n1, n2 in product(ch1, ch2)), \
                (v1, v2, constraint, heur.constraint_placement[constraint], ch1, ch2)



class TestTileBased(TestCase):

    def test1(self):
        h = TilePlacementHeuristic([], nx.Graph(), [])
        h.run()

    def test2(self):
        cs = [Constraint([[1, 2, 3], [4, 5]]), Constraint([[2, 3, 4], [5, 6]])]
        s = 3
        chs, g, orig = load_chimera(s)
        h = TilePlacementHeuristic(cs, g, chs)
        h.run()
        xdict = expand_solution(g, h.chains, orig)
        show_result(s, xdict)

    def test3(self):
        with open(dirname(__file__) + "/../simple60.cnf") as f:
            cnf = (parse_cnf(f))
        cnf = [mapl(lambda x: x // 2, clause) for clause in cnf[:130]]
        cs = list(cnf_to_constraints(cnf, max(max(x) for x in cnf)))
        s = 16
        chs, g, orig = load_chimera(s)
        h = TilePlacementHeuristic(cs, g, chs)
        print_(h.run(stop_first=True))
        for c, t in iteritems(h.constraint_placement):
            print_(c.tile, t)
        print_(repr(h.chains))
        test_result(g, cs, h)
        xdict = expand_solution(g, h.chains, orig)
        show_result(s, xdict)

    def test4(self):
        with open(dirname(__file__) + "/../simple60.cnf") as f:
            cnf = (parse_cnf(f))
        cnf = [mapl(lambda x: x // 6, clause) for clause in cnf[:50]]
        cs = list(cnf_to_constraints(cnf, max(max(x) for x in cnf)))
        s = 8
        chs, g, orig = load_chimera(s)
        h = TilePlacementHeuristic(cs, g, chs)
        print_(h.run(stop_first=True))
        for c, t in iteritems(h.constraint_placement):
            print_(c.tile, t)
        print_(repr(h.chains))
        test_result(g, cs, h)
        xdict = expand_solution(g, h.chains, orig)
        show_result(s, xdict)

    def test_par(self):
        with open(dirname(__file__) + "/../simple60.cnf") as f:
            cnf = (parse_cnf(f))
        cnf = [mapl(lambda x: x // 6, clause) for clause in cnf[:50]]
        cs = list(cnf_to_constraints(cnf, max(max(x) for x in cnf)))
        s = 16
        chs, g, orig = load_chimera(s)
        h = ParallelPlacementHeuristic(cs, g, chs)
        pool = Pool()
        print_(h.par_run(pool, stop_first=False))
        for c, t in iteritems(h.constraint_placement):
            print_(c.tile, t)
        print_(repr(h.chains))
        test_result(g, cs, h)
        xdict = expand_solution(g, h.chains, orig)
        show_result(s, xdict)

    def test_sgen(self):
        with open(dirname(__file__) + "/../../pysgen/allbenchmarks/sgen-twoinfour-s80-g4-0.bench") as f:
            cnf = (parse_2in4(f))
        nvars = max(max(x) for clause in cnf for x in clause)
        ancilla = nvars + 1
        for clause in cnf:
            clause[0].append(ancilla)
            clause[1].append(ancilla + 1)
            ancilla += 2
        cs = [Constraint(x) for x in cnf]

        choices, quotient_graph, original_graph = load_chimera(16)
        #quotient_graph, choices = g, chs
        pool = Pool()
        h = ParallelPlacementHeuristic(cs, quotient_graph, choices)
        print (h.par_run(pool, stop_first=True))
        print (expand_solution(quotient_graph, h.chains, original_graph))

