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

import numpy as np
from scipy.spatial import ConvexHull
def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    return lower[:-1] + upper[:-1]


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = np.array(convex_hull(points))

    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
#     rotations = np.vstack([
#         np.cos(angles),
#         -np.sin(angles),
#         np.sin(angles),
#         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    return x1, x2, y1, y2, r

def scale_on_arch(positions, archGraph):
    # for each placed problemedge pick closest archgraph, after proper scaling


    x1,x2,y1,y2,r = minimum_bounding_rectangle([x[1:] for x in positions])
    positions  = [(e,) + tuple(np.dot(r, (x,y))) for e,x,y in positions]

    xmin, xmax = min(positions, key=lambda x: x[1])[1], max(positions, key=lambda x: x[1])[1]
    ymin, ymax = min(positions, key=lambda x: x[2])[2], max(positions, key=lambda x: x[2])[2]


    anodes = {n:(d['x'], d['y']) for n,d in archGraph.nodes_iter(data=True)}
    amin = min(anodes.itervalues(), key=lambda x: x[0])[0], min(anodes.itervalues(), key=lambda x: x[1])[1]
    amax = max(anodes.itervalues(), key=lambda x: x[0])[0], max(anodes.itervalues(), key=lambda x: x[1])[1]
    awidth, aheight = (amax[0]-amin[0]), (amax[1]-amin[1])

    scaledpos = [(e, (x - xmin)/(xmax-xmin)*awidth+amin[0], (y-ymin)/(ymax-ymin)*aheight+amin[1]) for e, x, y in positions]
    return scaledpos


def ordered_choices(scaledpos, archGraph):
    anodes = {n: (d['x'], d['y']) for n, d in archGraph.nodes_iter(data=True)}
    archpos = [((anodes[e[0]][0] + anodes[e[1]][0]) / 2, (anodes[e[0]][1] + anodes[e[1]][1]) / 2, e) for e in
               archGraph.edges_iter()]
    ret = dict()
    for e, x, y in scaledpos:
        def distance(ae):
            ax, ay, e = ae
            return (x - ax) ** 2 + (y - ay) ** 2

        match = sorted(archpos, key=distance)
        ret[e] = [ae for (_,_,ae) in match[:24]]
    return ret

def match_closest(positions, archGraph):
    scaledpos = scale_on_arch(positions, archGraph)
    anodes = {n:(d['x'], d['y']) for n,d in archGraph.nodes_iter(data=True)}
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
