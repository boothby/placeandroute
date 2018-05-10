from multiprocessing import Pool
from unittest import TestCase

from os.path import dirname

from six import iteritems, print_

from placeandroute.tilebased.heuristic import TilePlacementHeuristic, Constraint
from placeandroute.tilebased.chimera_tiles import chimeratiles, expand_solution
from placeandroute.problemgraph import parse_cnf
import matplotlib

from placeandroute.tilebased.parallel import ParallelPlacementHeuristic

matplotlib.verbose = True # workaround for pycharm
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import networkx as nx
import dwave_networkx as dwnx
from itertools import product, cycle


def create(w, l):
    ret =  dwnx.chimera_graph(w,l)
    return ret, dwnx.chimera_layout(ret)

def cnf_to_constraints(clauses, num_vars):
    ancilla = num_vars + 1
    for clause in clauses:
        yield Constraint([map(abs, clause[:2]),[abs(clause[2]), ancilla]])
        ancilla += 1


def show_result(s, g, h):
    cg, layout = create(s, s)
    xdict = expand_solution(g, h.chains, cg)
    x = nx.Graph()
    for k, v in iteritems(xdict):
        x.add_node(k, mapto=v)
    #interactive_embeddingview(x, s, False)
    color = cycle(get_cmap("tab20").colors)
    nx.draw(cg, layout, node_color="gray", edge_color="gray")
    for k,vs in iteritems(xdict):
        col = next(color)
        subcg = cg.subgraph(vs)
        nx.draw(subcg, layout, node_color=col, edge_color=[col]*subcg.number_of_edges())
    plt.show()


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
            assert any(tile_graph.has_edge(n1,n2) for n1, n2 in product(ch1,ch2)), \
                (v1, v2, constraint, heur.constraint_placement[constraint], ch1, ch2)



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
        cnf = [map(lambda  x: x//2, clause) for clause in cnf[:130]]
        cs = list(cnf_to_constraints(cnf, max(max(x) for x in cnf)))
        s = 16
        g, chs = chimeratiles(s,s)
        h = TilePlacementHeuristic(cs, g, chs)
        print_(h.run(stop_first=True))
        for c, t in iteritems(h.constraint_placement):
            print_(c.tile, t)
        print_(repr(h.chains))
        test_result(g, cs, h)
        show_result(s, g, h)

    def test4(self):
        with open(dirname(__file__) + "/../simple60.cnf") as f:
            cnf = (parse_cnf(f))
        cnf = [map(lambda  x: x//6, clause) for clause in cnf[:50]]
        cs = list(cnf_to_constraints(cnf, max(max(x) for x in cnf)))
        s = 8
        g, chs = chimeratiles(s,s)
        h = TilePlacementHeuristic(cs, g, chs)
        print_(h.run())
        for c, t in iteritems(h.constraint_placement):
            print_(c.tile, t)
        print_(repr(h.chains))
        test_result(g, cs, h)
        show_result(s, g, h)

    def test_par(self):
        with open(dirname(__file__) + "/../simple60.cnf") as f:
            cnf = (parse_cnf(f))
        cnf = [map(lambda  x: x//6, clause) for clause in cnf[:50]]
        cs = list(cnf_to_constraints(cnf, max(max(x) for x in cnf)))
        s = 16
        g, chs = chimeratiles(s,s)
        h = ParallelPlacementHeuristic(cs, g, chs)
        pool = Pool(processes=4)
        print_(h.par_run(pool, stop_first=True))
        for c, t in iteritems(h.constraint_placement):
            print_(c.tile, t)
        print_(repr(h.chains))
        test_result(g, cs, h)
        show_result(s, g, h)

