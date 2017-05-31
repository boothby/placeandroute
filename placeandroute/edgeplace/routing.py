import networkx as nx
from typing import Any, Dict
from placeandroute.smtutils import Op, Symbol, SmtFormula
#min overfull
#given
# flux_in = 1 + overfull
# flux_out = flux_in + (0 /source/sink)
# overfull > 0
# 0 < flux < 1


def flow_conservation(formula, graph, commodities):
    def sum(it):
        return reduce(lambda a,b: a+b, it)
    for vertex in graph.nodes_iter():
        tot_in_flow = 0.0
        for comm in commodities:
            in_flow = sum(e["flow"][comm] for _,_,e in graph.in_edges_iter(vertex, data=True))
            tot_in_flow += in_flow
            out_flow = sum(e["flow"][comm] for _,_,e in graph.out_edges_iter(vertex, data=True))
            flow_consi = out_flow <= in_flow + graph.node[vertex]["flow_constant"][comm]
            formula.assert_(in_flow <= 1000 * graph.node[vertex]["vertex_assign"][comm])
            formula.assert_(flow_consi)
        flow_overfull = sum(graph.node[vertex]["vertex_assign"].values()) <= 1
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
        data["flow"] = dict()
        for comm in commodities:
            newsymb = Symbol("flow_{}_{}_{}".format(edge1+1, edge2+1, comm+1))
            formula.declare(newsymb, "Real")
            formula.assert_((newsymb >= 0.0))
            data["flow"][comm] = newsymb

    for vertex in graph.nodes_iter():
        vertex_assigns = dict()
        for comm in commodities:
            sym = Symbol("vertex_assign_{}_{}", vertex, comm)
            vertex_assigns[comm] = sym
            formula.declare(sym, "Int")
            formula.assert_((sym >=0) & (sym <= 1))
        graph.node[vertex]["vertex_assign"] = vertex_assigns





def do_routing(archGraph, problemGraph, nodeterminals):
    formula = SmtFormula()
    set_flow_constant(archGraph, nodeterminals)
    declare_flows(formula, archGraph, nodeterminals)
    flow_conservation(formula, archGraph, nodeterminals)
    return formula
