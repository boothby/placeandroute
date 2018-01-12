from unittest import TestCase

from chimerautils.display import interactive_embeddingview
from chimerautils.chimera import create
from os.path import dirname

from placeandroute.tilebased.heuristic import TilePlacementHeuristic, chimeratiles, Constraint
from placeandroute.problemgraph import parse_cnf
import networkx as nx
import os

def cnf_to_constraints(clauses, num_vars):
    ancilla = num_vars + 1
    for clause in clauses:
        yield Constraint([map(abs, clause[:2]),[abs(clause[2]), ancilla]])
        ancilla += 1

def expand_solution(g, h):
    x = nx.Graph()
    alreadymapped = set()
    cg, _ = create(16, 16)  # type: nx.Graph
    assert all(cg.has_edge(n1 * 4, n2 * 4) for n1, n2 in g.edges())
    for k, v in h.chains.iteritems():
        # single node case this is easy
        if len(v) == 1:
            nodegroup = next(iter(v))
            choice = range(nodegroup * 4, (nodegroup+1)*4)
            choice = filter(lambda  x: x not in alreadymapped, choice)[:1]
            alreadymapped.update(choice)
            x.add_node(k, mapto=choice)
            continue

        #long chain. rationale: build the subgraph of possible chains, then trim it
        #first filter already used nodes, then pick a connected component that has at least one choice
        # for each group, trim the choices slowly
        vs = []
        for nodegroup in v:
            choices = range(nodegroup * 4, (nodegroup + 1) * 4)
            choices = filter(lambda x: x not in alreadymapped, choices)
            assert choices
            vs.append(choices)
        allchoices = reduce(list.__add__, vs)
        sg = cg.subgraph(allchoices)  # type: nx.Graph
        chosen_sg = None
        for component in nx.connected_components(sg):
            #gcomponent = sg.subgraph(component)
            if all(any(n in component for n in choice) for choice in vs):
                chosen_sg = component
            else:
                for n in component:
                    allchoices.remove(n)
                    for v in vs:
                        if n in v: v.remove(n)
        sg = sg.subgraph(chosen_sg)
        sg = nx.minimum_spanning_tree(sg)
        assert nx.is_connected(sg)
        noprogress = 0
        while any(len(v) > 1 for v in vs):
            noprogress += 1
            for v in vs:
                if len(v) == 1: continue
                if not all(n in sg.nodes() for n in v):
                    raise Exception
                nn = filter(lambda n: sg.degree(n) == 1, v)
                if len(nn) == len(v): nn.pop(0)
                if nn:
                    for n in nn:
                        sg.remove_node(n)
                        allchoices.remove(n)
                        v.remove(n)
                    noprogress = 0
            if noprogress > 1000:
                for v in vs:
                    if len(v) > 1:
                        n = v.pop(0)
                        sg.remove_node(n)
                        allchoices.remove(n)
                        noprogress=0
                        break

        vs = [v[0] for v in vs]
        alreadymapped.update(vs)
        x.add_node(k, mapto=vs)
    return x

class TestTileBased(TestCase):
    def test1(self):
        h = TilePlacementHeuristic([],nx.Graph(), [])
        h.run()


    def test2(self):
        cs = [Constraint([[1,2,3],[4,5]]), Constraint([[2,3,4],[5,6]])]
        g, chs = chimeratiles(3,3)
        h = TilePlacementHeuristic(cs, g, chs)
        h.run()

    def test3(self):
        with open(dirname(__file__) + "/../simple60.cnf") as f:
            cnf = (parse_cnf(f))
        cnf = [map(lambda  x: x//2, clause) for clause in cnf[:20]]
        cs = list(cnf_to_constraints(cnf, max(max(x) for x in cnf)))
        g, chs = chimeratiles(16,16)
        h = TilePlacementHeuristic(cs, g, chs)
        print h.run()
        for c, t in h.placement.iteritems():
            print c.tile, t
        print repr(h.chains)
        os.environ["PATH"] += ":/home/svarotti/Drive/dwaveproj/projects/embeddingview"
        x = expand_solution(g, h)
        interactive_embeddingview(x, 16, False)

