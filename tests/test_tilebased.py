import logging
from itertools import product
from multiprocessing import Pool
from os.path import dirname
from unittest import TestCase

import dwave_networkx as dwnx
import networkx as nx
from six import iteritems, print_

from placeandroute.problemgraph import parse_cnf
from placeandroute.tilebased.chimera_tiles import chimeratiles, expand_solution, chimeratiles2, expand_solution2
from placeandroute.tilebased.heuristic import TilePlacementHeuristic, Constraint
from placeandroute.tilebased.parallel import ParallelPlacementHeuristic
from placeandroute.tilebased.utils import cnf_to_constraints, show_result


def test_result(tile_graph, cnf, heur):
    chains = heur.chains
    for constraint in cnf:
        assert heur.constraint_placement.has_key(constraint), heur.constraint_placement
        var_rows = constraint.tile
        for v1, v2 in product(*var_rows):
            if v1 == v2: continue
            ch1 = chains[v1]
            ch2 = chains[v2]
            assert nx.is_connected(tile_graph.subgraph(ch1))
            assert nx.is_connected(tile_graph.subgraph(ch2))
            assert any(tile_graph.has_edge(n1, n2) for n1, n2 in product(ch1, ch2)), \
                (v1, v2, constraint, heur.constraint_placement[constraint], ch1, ch2)


logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(asctime)s %(processName)s %(message)s')

class TestTileBased(TestCase):

    def test1(self):
        h = TilePlacementHeuristic([], nx.Graph(), [])
        h.run()

    def test2(self):
        cs = [Constraint([[1, 2, 3], [4, 5]]), Constraint([[2, 3, 4], [5, 6]])]
        s = 3
        g, chs = chimeratiles(s, s)
        h = TilePlacementHeuristic(cs, g, chs)
        h.run()
        xdict = expand_solution(g, h.chains, dwnx.chimera_graph(s))
        show_result(s, xdict)

    def test3(self):
        with open(dirname(__file__) + "/../simple60.cnf") as f:
            cnf = (parse_cnf(f))
        cnf = [map(lambda x: x // 2, clause) for clause in cnf[:130]]
        cs = list(cnf_to_constraints(cnf, max(max(x) for x in cnf)))
        s = 16
        g, chs = chimeratiles(s, s)
        h = TilePlacementHeuristic(cs, g, chs)
        print_(h.run(stop_first=True))
        for c, t in iteritems(h.constraint_placement):
            print_(c.tile, t)
        print_(repr(h.chains))
        test_result(g, cs, h)
        xdict = expand_solution2(g, h.chains, dwnx.chimera_graph(s))
        show_result(s, xdict)

    def test4(self):
        with open(dirname(__file__) + "/../simple60.cnf") as f:
            cnf = (parse_cnf(f))
        cnf = [map(lambda x: x // 6, clause) for clause in cnf[:50]]
        cs = list(cnf_to_constraints(cnf, max(max(x) for x in cnf)))
        s = 8
        g, chs = chimeratiles(s, s)
        h = TilePlacementHeuristic(cs, g, chs)
        print_(h.run(stop_first=True))
        for c, t in iteritems(h.constraint_placement):
            print_(c.tile, t)
        print_(repr(h.chains))
        test_result(g, cs, h)
        xdict = expand_solution2(g, h.chains, dwnx.chimera_graph(s))
        show_result(s, xdict)

    def test_par(self):
        with open(dirname(__file__) + "/../simple60.cnf") as f:
            cnf = (parse_cnf(f))
        #cnf = [map(lambda x: x // 6, clause) for clause in cnf[:50]]
        cs = list(cnf_to_constraints(cnf, max(max(x) for x in cnf)))
        s = 16
        g, chs = chimeratiles2(s, s)
        h = ParallelPlacementHeuristic(cs, g, chs)
        pool = Pool(processes=4)
        print_(h.par_run(pool, stop_first=False))
        for c, t in iteritems(h.constraint_placement):
            print_(c.tile, t)
        print_(repr(h.chains))
        test_result(g, cs, h)
        xdict = expand_solution2(g, h.chains, dwnx.chimera_graph(s))
        show_result(s, xdict)
