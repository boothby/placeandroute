from itertools import combinations

import networkx as nx

from placeandroute.smtutils import SmtFormula


def mapscore(problemGraph, archDiGraph):
    pass


def embed(problemGraph, archGraph):
    problemDiGraph = problemGraph.toDiGraph() # type: nx.DiGraph
    archDiGraph = archGraph.toDigraph()
    smtproblem = SmtFormula()
    equalvertexmap = dict()

    for i, v1, v2 in zip(range(0,2*len(problemGraph.edges),2), problemGraph.edges_iter()):
        smtproblem.declare("pa_{}".format(i), "Int")
        smtproblem.declare("pa_{}".format(i + 1), "Int")
        smtproblem.declare("pb_{}".format(i), "Int")
        smtproblem.declare("pb_{}".format(i + 1), "Int")

        # p1[i] = p2[i+1] = v1
        # (assert (= pa_{} pb_{} {}) )
        smtproblem.assert_("(= pa_{} pb_{} {})".format(i, i+1, v1))
        # p2[i] = p1[i+1] = v2
        # (assert (= pa_{} pb_{} {}) )
        smtproblem.assert_("(= pb_{} pa_{} {})".format(i, i+1, v2))



    for i in range(_):
        # m1[i] = m2[i+1]
        # m2[i] = m1[i+1]
        for j in range(_):
            # xx collect all eq m1[i] == m1[j] => p1[i] == p1[j]
        pass

    #declare cost
    usedsum = ""
    for v in problemGraph.nodes_iter():
        # count all m1[i] where p1[i] = v
        for ma, mb in combinations(equalvertexmap[v]):
            usedsum += " (ite (= {} {}) 1 0)".format(ma, mb)
    smtproblem.assert_("(minimize (+ {})".format(usedsum))
    #minimize counts
    


