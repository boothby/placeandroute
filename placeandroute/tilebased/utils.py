from itertools import cycle

import dwave_networkx as dwnx
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap

from placeandroute.tilebased.heuristic import Constraint


def create(w):
    ret = dwnx.chimera_graph(w, w)
    return ret, dwnx.chimera_layout(ret)


def cnf_to_constraints(clauses, num_vars):
    """from each 3-cnf clause, generate a Constraint with a fresh ancilla variable"""
    ancilla = num_vars + 1
    for clause in clauses:
        #                 first two vars        third var + ancilla
        clause = list(abs(l) for l in clause)
        assert len(clause) == 3, clause
        c  = Constraint()
        for literal in clause:
            c.add_possible_placement([[ancilla, literal],[l for l in clause if l != literal]])
        yield c
        ancilla += 1


def show_result(cg, xdict, layout=dwnx.chimera_layout):
    """Display a Chimera embedding using matplotlib"""

    layout = layout(cg)
    color = cycle(get_cmap("tab20").colors)
    nx.draw(cg, layout, node_color="gray", edge_color="gray")
    for k, vs in xdict.items():
        col = next(color)
        subcg = cg.subgraph(vs)
        nx.draw(subcg, layout, node_color=[col] * subcg.number_of_nodes(), edge_color=[col] * subcg.number_of_edges())
    plt.savefig("result.png")
