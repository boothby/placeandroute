from itertools import combinations
from collections import defaultdict
import networkx as nx

from placeandroute.smtutils import SmtFormula, Op, Symbol


def mapscore(problemGraph, archDiGraph):
    pass


def place_edges(problemGraph, archGraph):
    problemDiGraph = problemGraph.to_directed() # type: nx.DiGraph
    archDiGraph = archGraph.to_directed()
    smtproblem = SmtFormula()
    equalvertexmap = defaultdict(list)
    ma, mb = Symbol("arch1"), Symbol("arch2")
    smtproblem.declare(ma, "Int", args="(Int)")
    smtproblem.declare(mb, "Int", args="(Int)")


    # (ma,mb) indexed arch edges
    smtproblem.assert_(ma(0) == mb(0) == 0)
    for i, (n1,n2) in enumerate(archGraph.edges_iter(),start=1):
        smtproblem.assert_(ma(i) == mb(-i) == n1)
        smtproblem.assert_(mb(i) == ma(-i) == n2)

    #(pa, pb) indexed problem edges
    #map(i) map problem to arch
    for i, (v1, v2) in enumerate(problemGraph.edges_iter(), start=1):
        pa = Symbol("pa_{}", i)
        pb = Symbol("pb_{}", i)
        pmap = Symbol("map_{}", i)

        smtproblem.declare(pa, "Int")
        smtproblem.declare(pb, "Int")
        smtproblem.declare(pmap, "Int")

        smtproblem.assert_(pa == v1)
        smtproblem.assert_(pb == v2)

        equalvertexmap[v1].append(pmap)
        equalvertexmap[v2].append(-pmap)
        smtproblem.assert_(  (pmap <= archGraph.number_of_edges()) & (pmap != 0)
                           & (pmap >= -archGraph.number_of_edges()))


        # (ma(map(i)) = ma(map(j))) -> (pa(i) == pa(j))
        # and with other node
        for j in xrange(1, i):
            mapj = Symbol("map_{}", j)
            paj = Symbol("pa_{}", j)
            smtproblem.assert_(Op("=>", ma(pmap) == ma(mapj), pa == paj))
            smtproblem.assert_(Op("=>", ma(pmap) == ma(-mapj), pb == paj))

    cost = dict()
    #declare cost
    for v, mapsto in equalvertexmap.iteritems():
        cost[v] = []
        for mapv in mapsto:
            pass#costv.append(Op("ite", ))


    

    return smtproblem
