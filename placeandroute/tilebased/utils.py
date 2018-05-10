from itertools import cycle

import dwave_networkx as dwnx
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
from six import iteritems

from placeandroute.tilebased.heuristic import Constraint


def create(w):
    ret = dwnx.chimera_graph(w, w)
    return ret, dwnx.chimera_layout(ret)


def cnf_to_constraints(clauses, num_vars):
    """from each 3-cnf clause, generate a Constraint with a fresh ancilla variable"""
    ancilla = num_vars + 1
    for clause in clauses:
        #                 first two vars        third var + ancilla
        yield Constraint([map(abs, clause[:2]), [abs(clause[2]), ancilla]])
        ancilla += 1


def show_result(s, xdict):
    """Display a Chimera embedding using matplotlib"""
    cg, layout = create(s)

    color = cycle(get_cmap("tab20").colors)
    nx.draw(cg, layout, node_color="gray", edge_color="gray")
    for k, vs in iteritems(xdict):
        col = next(color)
        subcg = cg.subgraph(vs)
        nx.draw(subcg, layout, node_color=col, edge_color=[col] * subcg.number_of_edges())
    plt.show()
