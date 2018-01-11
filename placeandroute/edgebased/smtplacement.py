from collections import defaultdict

import networkx as nx
from smtutils.formula import SmtFormula, Op, Symbol


def mapscore(problemGraph, archDiGraph):
    pass


def place_edges(problemGraph, archGraph):
    problemDiGraph = problemGraph.to_directed() # type: nx.DiGraph
    archDiGraph = archGraph.to_directed()
    smtproblem = SmtFormula()
    equalvertexmap = defaultdict(list)
    ma, mb = Symbol("arch1"), Symbol("arch2")
    terminalof = Symbol("terminalof")
    sourceof = Symbol("sourceof")
    smtproblem.declare(ma, "Int", args="(Int)")
    smtproblem.declare(mb, "Int", args="(Int)")
    smtproblem.declare(terminalof, "Int", args="(Int)")
    smtproblem.declare(sourceof, "Int", args="(Int)")


    # (ma,mb) indexed arch edges
    smtproblem.assert_((ma(0) == 0) & ( mb(0) == 0))
    for i, (n1,n2) in enumerate(archGraph.edges_iter(),start=1):
        smtproblem.assert_((ma(i) == mb(-i)) & (ma(i) == n1+1))
        smtproblem.assert_((mb(i) == ma(-i)) & (mb(i) == n2+1))

    #(pa, pb) indexed problem edges
    #map(i) map problem to arch
    for i, (v1, v2) in enumerate(problemGraph.edges_iter(), start=1):
        pa = Symbol("pa_{}", i)
        pb = Symbol("pb_{}", i)
        pmap = Symbol("map_{}", i)

        smtproblem.declare(pa, "Int")
        smtproblem.declare(pb, "Int")
        smtproblem.declare(pmap, "Int")

        smtproblem.assert_(pa == v1+1)
        smtproblem.assert_(pb == v2+1)

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


        # define function terminalof(archNode) -> problemNode
        smtproblem.assert_(terminalof(ma(pmap)) == pa)
        smtproblem.assert_(terminalof(mb(pmap)) == pb)

        #define function sourceof(problemNode) -> archNode == min { an : terminalof(an) == pn}
        smtproblem.assert_(sourceof(pa) <= ma(pmap))
        smtproblem.assert_(terminalof(sourceof(pa)) == pa)
        smtproblem.assert_(sourceof(pb) <= mb(pmap))
        smtproblem.assert_(terminalof(sourceof(pb)) == pb)

    set_flow_constant(smtproblem, archDiGraph, problemGraph.nodes(), sourceof, terminalof)
    from placeandroute.routing.smtbased import  declare_flows, flow_conservation
    declare_flows(smtproblem, archDiGraph, problemGraph.nodes())
    flow_conservation(smtproblem, archDiGraph, problemGraph.nodes())

    return smtproblem

def set_flow_constant(smtproblem, graph, commodities, sourceof, terminalof):
    #hinum = dict()
    #for comm in commodities:
    #    hnum = 0
    #    for vertex in graph.nodes_iter():
    #        hnum += Op("ite", terminalof(vertex+1) == comm+1, 1, 0)
    #    hinum[comm] = Symbol("hinum_{}", comm+1)
    #    smtproblem.declare(hinum[comm], "Int")
    #    smtproblem.assert_(hinum[comm] == hnum)

    for vertex in graph.nodes_iter():
        graph.node[vertex]["flow_constant"] = dict()
        for comm in commodities:
            flow_constant = Op("ite", (terminalof(vertex+1) == comm+1),
                               (Op("ite", (sourceof(comm+1) == vertex+1),
                               #    (hinum[comm] - 1), (-1))),
                                   (1000.0), (-1.0))),
                               (0.0))
            graph.node[vertex]["flow_constant"][comm] = flow_constant