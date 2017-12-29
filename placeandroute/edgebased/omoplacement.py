from __future__ import division
from pyomo.environ import *
from math import sqrt
from typing import List, Tuple
import networkx as nx


def place_edges(graph):
    # type: (nx.Graph) -> List[Tuple[Any, float,float]]
    edges = graph.edges()

    #def sum(it):
    #    return reduce(lambda a, b: a + b, it)

    edgetoindex = {k: v for v, k in enumerate(edges)}
    edgetoindex.update({(b, a): v for v, (a, b) in enumerate(edges)})
    nodes = graph.nodes()
    nodetoindex = {k: v for v, k in enumerate(nodes)}

    optmod = ConcreteModel()

    optmod.x = Var(range(graph.number_of_edges()), domain=Reals, initialize= (1/sqrt(len(edges))))
    optmod.y = Var(range(graph.number_of_edges()), domain=Reals, initialize= (1/sqrt(len(edges))))

    optmod.nx = Var(range(graph.number_of_nodes()), domain=Reals)
    optmod.ny = Var(range(graph.number_of_nodes()), domain=Reals)
    optmod.v = Var(range(graph.number_of_nodes()), domain=NonNegativeReals)

    optmod.OBJ = Objective(expr=sum(optmod.v[i] for i in range(graph.number_of_nodes())))

    optmod.clist = ConstraintList()
    optmod.sparsex = Constraint(expr=sum(optmod.x[i]**2 for i in range(len(edges))) == 1)
    optmod.sparsex2 = Constraint(expr=sum(optmod.x[i] for i in range(len(edges))) == 0)

    optmod.sparsey = Constraint(expr=sum(optmod.y[i]**2 for i in range(len(edges)))  == 1)
    optmod.sparsey2 = Constraint(expr=sum(optmod.y[i] for i in range(len(edges))) == 0)



    for i, node in enumerate(nodes):
        optmod.clist.add((sum(optmod.x[edgetoindex[node, nn]] / graph.degree(node) for nn in graph.neighbors_iter(node))
                         ) == optmod.nx[i])
        optmod.clist.add(
            (sum(optmod.y[edgetoindex[node, nn]] / graph.degree(node) for nn in graph.neighbors_iter(node))
            ) == optmod.ny[i])
        optmod.clist.add(
            (sum((optmod.nx[i] - optmod.x[edgetoindex[node, nn]]) **2 for nn in
                 graph.neighbors_iter(node))
             +
             sum((optmod.ny[i] - optmod.y[edgetoindex[node, nn]]) **2 for nn in
                 graph.neighbors_iter(node))
             ) == optmod.v[i]
        )
    # for e in optmod.clist.itervalues():
    #    e = e._body
    #    e._const = None
    #optmod.pprint()

    solver = SolverFactory('ipopt')
    solution = solver.solve(optmod)
    optmod.solutions.load_from(solution)
    #print "DISPLAY"
    #optmod.display()

    return [(e, optmod.x[i].value, optmod.y[i].value) for i,e  in enumerate(edges)]


def match_closest(positions, archGraph):
    # for each placed problemedge pick closest archgraph, after proper scaling
    xmin, xmax = min(positions, key=lambda x: x[1])[1], max(positions, key=lambda x: x[1])[1]
    ymin, ymax = min(positions, key=lambda x: x[2])[2], max(positions, key=lambda x: x[2])[2]

    anodes = {n:(d['x'], d['y']) for n,d in archGraph.nodes_iter(data=True)}
    amin = min(anodes.itervalues(), key=lambda x: x[0])[0], min(anodes.itervalues(), key=lambda x: x[1])[1]
    amax = max(anodes.itervalues(), key=lambda x: x[0])[0], max(anodes.itervalues(), key=lambda x: x[1])[1]
    awidth, aheight = (amax[0]-amin[0]), (amax[1]-amin[1])

    scaledpos = [(e, (x - xmin)/xmax*awidth+amin[0], (y-ymin)/ymax*aheight+amin[1]) for e, x, y in positions]

    archpos = [((anodes[e[0]][0]+anodes[e[1]][0])/2, (anodes[e[0]][1] + anodes[e[1]][1])/2, e) for e in archGraph.edges_iter()]
    ret = dict()
    for e, x,y in scaledpos:
        def distance(ae):
            ax, ay, e = ae
            return (x-ax)**2 + (y-ay)**2
        match = min(archpos, key=distance)
        archpos.remove(match)
        ret[e] = match[2]
    return ret
