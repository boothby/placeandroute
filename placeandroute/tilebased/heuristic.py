from math import exp

import math
from typing import List, Any, Dict
import networkx as nx

from placeandroute.tilebased.tactics import RandomInitTactic, BFSInitTactic, RipRerouteTactic, RerouteTactic
from ..routing.bonnheuristic import bounded_exp

class Constraint(object):
    """Class representing a constraint, essentially just a container for the variable mapping.
    """
    def __init__(self, vars):
        # type: (List[List[Any]]) -> None
        """Initializes the variable mapping.
        Currently the variable mapping is represented as a list of list of variables. Each inner list represents a set
        of variables that can be placed in any order. Considering Chimera, the mapping consists in two lists, each
        list containing the variables that go in a 4 qubit row. Todo: add multiple alternative mappings"""
        self.tile = tuple(tuple(varrow) for varrow in vars)

    def __eq__(self, other):
        return self.tile == other.tile

    def __hash__(self):
        return hash(self.tile)

class TilePlacementHeuristic(object):
    """Heuristic constraint placer. Tries to decrease overused qubits by ripping and rerouting"""
    def __init__(self, constraints, archGraph, choices):
        # type: (List[Constraint], nx.Graph, List[List[Any]]) -> None
        self.constraints = constraints
        self.arch = archGraph
        self.constraint_placement = dict()
        self.chains = {}
        self.choices = choices
        self.coeff = math.log(sum(d["capacity"] for _, d in archGraph.nodes(data=True)))  # high to discourage overlap
        print("coeff", exp(self.coeff))

        # tactic choices
        self._init_tactics = [BFSInitTactic, RandomInitTactic] * 2 + [RandomInitTactic] * 1
        self._improve_tactics = [RipRerouteTactic.default(), RerouteTactic] * 50


    def run(self, stop_first=False):
        """Run the place and route heuristic. Iterate between tactics until a solution is found"""
        self.clear_best()
        found = False
        for init_tactic in self._init_tactics:
            init_tactic.run_on(self)
            print("Initialized, score is: {}, overlapping: {}".format(self.score(), self.get_overlapping()))
            if self.is_valid_embedding():
                self.save_best()
                if stop_first:
                    self.restore_best()
                    return True
                found = True
            for improve_tactic in self._improve_tactics:
                improve_tactic(self).run()
                print("Score improved to: {}, overlapping: {}".format(self.score(), self.get_overlapping()))
                if self.is_valid_embedding():
                    self.save_best()
                    if stop_first:
                        self.restore_best()
                        return True
                    found = True

        self.restore_best()
        return found

    def clear_best(self):
        self._best_score = None
        self._best_plc = (None, None)

    def save_best(self):
        score = self.score()
        if self._best_score is None or score < self._best_score:
            self._best_score = score
            self._best_plc = (self.constraint_placement.copy(), self.chains.copy())

    def restore_best(self):
        (self.constraint_placement, self.chains) = self._best_plc


    def is_valid_embedding(self):
        # type: () -> bool
        """Check if the current embedding is valid"""
        return self.get_overlapping() == 0

    def get_overlapping(self):
        return sum(max(0, data['usage'] - data['capacity']) for _,data in self.arch.nodes(data=True))


    def scores(self):
        # type: () -> Dict[Any, float]
        """Return usage scores for the current qubits. Currently (e**overusage -1)/(e -1)"""
        return {n:
                #    (exp(max(0, data['usage'] - data['capacity']))-1)/(math.e -1)
                    data["usage"] + bounded_exp(
                        self.coeff * max(0, data['usage'] - data['capacity'])) - 1
                for n, data in self.arch.nodes(data=True)}


    def score(self):
        return sum(self.scores().itervalues())


    def var_placement(self):
        # type: () -> Dict[Any,List[Any]]
        """Return variable placement (from constraint placement)"""
        placement = dict()
        for constraint, tile in self.constraint_placement.iteritems():
            for pnodes, anode in zip(constraint.tile, tile):
                for pnode in pnodes:
                    if pnode not in placement:
                        placement[pnode] = []
                    placement[pnode].append(anode)
        return placement


    def fix_usage(p):
        for node, data in p.arch.nodes(data=True):
            data["usage"] = 0
        for nodes in p.chains.itervalues():
            for node in nodes:
                p.arch.nodes[node]["usage"] += 1