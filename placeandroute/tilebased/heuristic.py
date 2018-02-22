from math import exp

import math
from typing import List, Any, Dict
import networkx as nx

from placeandroute.tilebased.tactics import RandomInitTactic, BFSInitTactic, IterPickTactic, CostlyTilePickTactic, \
    ChainsFindPlaceTactic, RerouteInsertTactic, IncrementalRerouteInsertTactic


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
        self.best_found = (None,)

        # tactic choices (to be refactored)
        # InitTactic and ImproveTactic, using the ones below
        self._init_tactics = [BFSInitTactic, RandomInitTactic] * 3 + [RandomInitTactic] * 4
        self._pick_tactics = [CostlyTilePickTactic, IterPickTactic, CostlyTilePickTactic]
        self._find_tactic = ChainsFindPlaceTactic(self)
        #self._insert_tactic = RerouteInsertTactic(self)
        #self._insert_tactic = SimpleInsertTactic(self)
        self._insert_tactic = IncrementalRerouteInsertTactic(self)


    def find_best_place(self, constraint):
        # this method will move soon to a tactic class
        return self._find_tactic.find_best_place(constraint)


    def run(self):
        """Run the place and route heuristic. Iterate between tactics until a solution is found"""
        for TClass in self._init_tactics:
            init_tactic = TClass(self)
            if not init_tactic.init_placement():
                return False
            if self.do_try():
                return True
        return False

    def do_try(self):
        # try to improve result ripping and rerouting or rerouting everything. Probably will be refactored into .run()
        effort = 10.0
        rare_reroute = 0
        for TClass in self._pick_tactics:
            tactic = TClass(self)
            no_improvement = 0
            while no_improvement <= tactic.max_no_improvement:
                if self.is_valid_embedding():
                    return True

                score = self.score()
                print(int(score), [self.constraint_placement[c] for c in self.constraints])
                if self.save_best(if_score=score):
                    no_improvement = 0
                else:
                    no_improvement += 1
                    rare_reroute += 1
                if rare_reroute >= 20:
                    rare_reroute = 0
                    effort += 5.0
                    RerouteInsertTactic(self).do_routing(int(effort))
                else:
                    remove_choice = tactic.pick_worst()
                    self.remove_tile(remove_choice)
                    #print self.score()
                    best_place = self.find_best_place(remove_choice)
                    self.insert_tile(remove_choice, best_place)

        return self.is_valid_embedding()

    def save_best(self, if_score):
        # type: (float) -> bool
        """Check if the current result has been the best so far"""
        if self.best_found[0] is None or if_score < self.best_found[0]:
            self.best_found = (if_score, self.constraint_placement, self.chains)
            return True
        else:
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

    def remove_tile(self, constr, clean_chains=True):
        # remove a constraint from the current embedding. Todo: put into a tactic
        old =  self.constraint_placement[constr]
        del self.constraint_placement[constr]

        if clean_chains: # todo: always True

            var_placement = self.var_placement()
            for var_row, node in zip(constr.tile, old):
                for var in var_row:
                    if var not in self.chains or node not in self.chains[var]:
                        continue # should be hit only when var has been deleted from the previous row

                    if var not in var_placement:
                        # the deleted tile was the only one containing this variables, clear the whole chain
                        for deleted_node in self.chains[var]:
                            self.arch.nodes[deleted_node]['usage'] -= 1
                        del self.chains[var]
                        continue

                    chaintree = nx.Graph(self.arch.subgraph(self.chains[var]))
                    leaf = node
                    # remove only unused qubits from the chain (leaf in the tree that do not belong to other constraints
                    while chaintree.degree(leaf) == 1 and leaf not in var_placement[var]:
                        newleaf = next(chaintree.neighbors(leaf))
                        chaintree.remove_node(leaf)
                        self.chains[var] = self.chains[var].difference({leaf})
                        self.arch.nodes[leaf]['usage'] -= 1
                        leaf = newleaf


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

    def insert_tile(self, constraint, tile):
        # type: (Constraint, List[List[Any]]) -> None
        # todo: move into tactic
        self._insert_tactic.insert_tile(constraint, tile)


