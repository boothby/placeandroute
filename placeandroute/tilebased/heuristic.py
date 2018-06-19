import math

import networkx as nx
from six import itervalues, iteritems
from typing import List, Any, Dict

from placeandroute.tilebased.tactics import RandomInitTactic, BFSInitTactic, RipRerouteTactic, RerouteTactic
from ..routing.bonnheuristic import bounded_exp
import logging


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

default_init_tactics = [BFSInitTactic.default(), RandomInitTactic.default()] * 2 + [RandomInitTactic.default()] * 1
default_improve_tactics = [RipRerouteTactic.repeated(), RerouteTactic.default()] * 50

class TilePlacementHeuristic(object):
    """Heuristic constraint placer. Tries to decrease overused qubits by ripping and rerouting"""

    def __init__(self, constraints, archGraph, choices, init_tactics=default_init_tactics,
                 improve_tactics=default_improve_tactics):
        # type: (List[Constraint], nx.Graph, List[List[Any]]) -> None
        self.constraints = constraints
        self.arch = archGraph
        self.constraint_placement = dict()
        self.chains = {}
        self.choices = choices
        self.coeff = math.log(sum(d["capacity"] for _, d in archGraph.nodes(data=True)))  # high to discourage overlap
        self.clear_best()

        # tactic choices
        self._init_tactics = init_tactics
        self._improve_tactics = improve_tactics


    def initialize_tactics(self):
        logging.info("Tactics summary")
        logging.info("Initialization")
        init_tactic_factories = self._init_tactics
        self._init_tactics = []
        for tacticFactory in init_tactic_factories:
            tactic = tacticFactory.create(self)
            self._init_tactics.append(tactic)
            logging.info("%s", tactic)

        improve_tactic_factories = self._improve_tactics
        self._improve_tactics = []
        logging.info("Improvement")
        for tacticFactory in improve_tactic_factories:
            tactic = tacticFactory.create(self)
            self._improve_tactics.append(tactic)
            logging.info("%s", tactic)

    def run(self, stop_first=False):
        """Run the place and route heuristic. Iterate between tactics until a solution is found"""
        self.initialize_tactics()
        self.clear_best()
        found = False
        logging.info("P&R start")
        for init_tactic in self._init_tactics:
            init_tactic.run()
            logging.info("Initialized, score is: %e, overlapping: %d",self.score(),self.get_overlapping())
            if self.is_valid_embedding():
                self.save_best()
                if stop_first:
                    self.restore_best()
                    return True
                found = True
            for improve_tactic in self._improve_tactics:
                improve_tactic.run()
                logging.info("New score: %e, overlapping: %d",self.score(), self.get_overlapping())
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
        return sum(max(0, data['usage'] - data['capacity']) for _, data in self.arch.nodes(data=True))

    def scores(self):
        # type: () -> Dict[Any, float]
        """Return usage scores for the current qubits. Currently (e**overusage -1)/(e -1)"""
        return {n:
                #    (exp(max(0, data['usage'] - data['capacity']))-1)/(math.e -1)
                    data["usage"] + bounded_exp(
                        self.coeff * max(0, data['usage'] - data['capacity'])) - 1
                for n, data in self.arch.nodes(data=True)}

    def score(self):
        return sum(itervalues(self.scores()))

    def var_placement(self):
        # type: () -> Dict[Any,List[Any]]
        """Return variable placement (from constraint placement)"""
        placement = dict()
        for constraint, tile in iteritems(self.constraint_placement):
            for pnodes, anode in zip(constraint.tile, tile):
                for pnode in pnodes:
                    if pnode not in placement:
                        placement[pnode] = []
                    placement[pnode].append(anode)
        return placement

    def fix_usage(p):
        for node, data in p.arch.nodes(data=True):
            data["usage"] = 0
        for nodes in itervalues(p.chains):
            for node in nodes:
                p.arch.nodes[node]["usage"] += 1
