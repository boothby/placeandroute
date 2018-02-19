from unittest import TestCase

from chimerautils.chimera import create
from chimerautils.display import interactive_embeddingview
from os.path import dirname

from placeandroute.tilebased.heuristic import TilePlacementHeuristic, Constraint
from placeandroute.tilebased.chimera_tiles import chimeratiles, expand_solution
from placeandroute.problemgraph import parse_cnf
import networkx as nx
import os
from itertools import product

def cnf_to_constraints(clauses, num_vars):
    ancilla = num_vars + 1
    for clause in clauses:
        yield Constraint([map(abs, clause[:2]),[abs(clause[2]), ancilla]])
        ancilla += 1


def show_result(s, g, h):
    os.environ["PATH"] += ":/home/svarotti/Drive/dwaveproj/projects/embeddingview"
    cg, _ = create(s, s)
    xdict = expand_solution(g, h.chains, cg)
    x = nx.Graph()
    for k, v in xdict.iteritems():
        x.add_node(k, mapto=v)
    interactive_embeddingview(x, s, False)

def test_result(tile_graph, cnf, heur):
    chains = heur.chains
    for constraint in cnf:
        assert heur.placement.has_key(constraint)
        var_rows = constraint.tile
        for v1, v2 in product(*var_rows):
            if v1 == v2: continue
            ch1 = chains[v1]
            ch2 = chains[v2]
            assert nx.is_connected(tile_graph.subgraph(ch1))
            assert nx.is_connected(tile_graph.subgraph(ch2))
            assert any(tile_graph.has_edge(n1,n2) for n1, n2 in product(ch1,ch2)), (v1, v2, constraint, heur.placement[constraint], ch1, ch2)



class TestTileBased(TestCase):
    def test1(self):
        h = TilePlacementHeuristic([],nx.Graph(), [])
        h.run()


    def test2(self):
        cs = [Constraint([[1,2,3],[4,5]]), Constraint([[2,3,4],[5,6]])]
        s = 3
        g, chs = chimeratiles(s,s)
        h = TilePlacementHeuristic(cs, g, chs)
        h.run()
        show_result(s,g, h)


    def test3(self):
        with open(dirname(__file__) + "/../simple60.cnf") as f:
            cnf = (parse_cnf(f))
        cnf = [map(lambda  x: x//3, clause) for clause in cnf[:90]]
        cs = list(cnf_to_constraints(cnf, max(max(x) for x in cnf)))
        s = 16
        g, chs = chimeratiles(s,s)
        h = TilePlacementHeuristic(cs, g, chs)
        print h.run()
        for c, t in h.placement.iteritems():
            print c.tile, t
        print repr(h.chains)
        show_result(s, g, h)

    def test4(self):
        with open(dirname(__file__) + "/../simple60.cnf") as f:
            cnf = (parse_cnf(f))
        cnf = [map(lambda  x: x//6, clause) for clause in cnf[:50]]
        cs = list(cnf_to_constraints(cnf, max(max(x) for x in cnf)))
        s = 8
        g, chs = chimeratiles(s,s)
        h = TilePlacementHeuristic(cs, g, chs)
        print h.run()
        for c, t in h.placement.iteritems():
            print c.tile, t
        print repr(h.chains)
        test_result(g, cs, h)
        show_result(s, g, h)


