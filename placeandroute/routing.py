import networkx as nx
from typing import Any, Dict
from smtutils import Op, Symbol, SmtFormula
#min overfull
#given
# flux_in = 1 + overfull
# flux_out = flux_in + (0 /source/sink)
# overfull > 0
# 0 < flux < 1


def flow_conservation(formula, graph, commodities):
    for vertex in graph.nodes_iter():
        tot_in_flow = 0
        for comm in commodities.iterkeys():
            in_flow = sum(e["flow"][comm] for _,e in graph.in_degree_iter(vertex))
            tot_in_flow += in_flow
            out_flow = sum(e["flow"][comm] for _,e in graph.out_degree_iter(vertex))
            flow_consi = out_flow == in_flow + graph.nodes[vertex]["flow_constant"][comm]
            formula.assert_(flow_consi)
        flow_overfull = tot_in_flow <= 1
        formula.assert_(flow_overfull)


def set_flow_constant(graph, commodities):
    for vertex in graph.nodes_iter():
        for comm, cdata in commodities.iteritems():
            if vertex == cdata["source"]:
                flow_constant = len(cdata["sinks"])
            elif vertex in cdata["sinks"]:
                flow_constant = -1
            else:
                flow_constant = 0
            graph.nodes[vertex]["flow_constant"][comm] = flow_constant


def declare_flows(formula, graph, commodities):
    for edge1,edge2, data in graph.edges_iter(data=True):
        for comm in commodities:
            newsymb = Symbol("flow_{}_{}_{}".format(edge1, edge2, comm))
            formula.declare(newsymb, "Real")
            formula.assert_((newsymb >= 0) & (newsymb <= 1))
            data["flow"][comm] = newsymb





def do_routing(archGraph, problemGraph, nodeterminals):
    formula = SmtFormula()
    set_flow_constant(archGraph, nodeterminals)
    declare_flows(formula, archGraph, nodeterminals)
    flow_conservation(formula, archGraph, nodeterminals)
