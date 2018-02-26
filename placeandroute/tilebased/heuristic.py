from math import exp

import math
from typing import List, Any, Dict
import networkx as nx

from placeandroute.tilebased.tactics import RandomInitTactic, BFSInitTactic, RipRerouteTactic, RerouteTactic


class Constraint(object):
    """Class representing a constraint, essentially just a container for the variable mapping.
    """
    def __init__(self, vars):
        # type: (List[List[Any]]) -> None
        """Initializes the variable mapping.
        Currently the variable mapping is represented as a list of list of variables. Each inner list represents a set
        of variables that can be placed in any order. Considering Chimera, the mapping consists in two lists, each
        list containing the variables that go in a 4 qubit row. Todo: add multiple alternative mappings"""
        self.tile = vars


class TilePlacementHeuristic(object):
    """Heuristic constraint placer. Tries to decrease overused qubits by ripping and rerouting"""
    def __init__(self, constraints, archGraph, choices):
        # type: (List[Constraint], nx.Graph, List[List[Any]]) -> None
        self.constraints = constraints
        self.arch = archGraph
        self.constraint_placement = dict()
        self.chains = {}
        self.choices = choices

        # tactic choices
        self._init_tactics = [BFSInitTactic, RandomInitTactic] * 3 + [RandomInitTactic] * 4
        self._improve_tactics = [RipRerouteTactic.default(), RerouteTactic] * 50


    def run(self):
        """Run the place and route heuristic. Iterate between tactics until a solution is found"""
        for init_tactic in self._init_tactics:
            init_tactic.run_on(self)
            print("Initialized, score is: {}".format(self.score()))
            if self.is_valid_embedding():
                return True
            for improve_tactic in self._improve_tactics:
                improve_tactic(self).run()
                print("Score improved to: {}".format(self.score()))
                if self.is_valid_embedding():
                    return True

        return False


    def is_valid_embedding(self):
        # type: () -> bool
        """Check if the current embedding is valid"""
        return all(data['usage'] <= data['capacity'] for _,data in self.arch.nodes(data=True))


    def scores(self):
        # type: () -> Dict[Any, float]
        """Return usage scores for the current qubits. Currently (e**overusage -1)/(e -1)"""
        return {n:(exp(max(0, data['usage'] - data['capacity']))-1)/(math.e -1) for n, data in self.arch.nodes(data=True)}


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