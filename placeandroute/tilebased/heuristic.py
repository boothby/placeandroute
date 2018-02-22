from math import exp

import math
from typing import List, Any
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
        of variables that can be placed in any order. Considering the chimera, the mapping consists in two lists
        containing the variables that go in a 4 qubit row. Todo: add multiple alternative mappings"""
        self.tile = vars


class TilePlacementHeuristic(object):
    """Heuristic constraint placer. Tries to decrease overused qubits by ripping and rerouting"""
    def __init__(self, constraints, archGraph, choices):
        # type: (List[Constraint], nx.Graph, List[List[Any]]) -> None
        self.constraints = constraints
        self.arch = archGraph
        self.placement = dict()
        self.chains = {}
        self.choices = choices
        self.best_found = (None,)
        self._init_tactics = [BFSInitTactic, RandomInitTactic] * 3 + [RandomInitTactic] * 4
        self._pick_tactics = [CostlyTilePickTactic, IterPickTactic, CostlyTilePickTactic]
        self._find_tactic = ChainsFindPlaceTactic(self)
        #self._insert_tactic = RerouteInsertTactic(self)
        #self._insert_tactic = SimpleInsertTactic(self)
        self._insert_tactic = IncrementalRerouteInsertTactic(self)


    def find_best_place(self, constraint):
        return self._find_tactic.find_best_place(constraint)

    def constraint_fit(self, choice, constraint):
        for (node, req) in zip(choice, constraint.tile):
            data =self.arch.nodes[node]

            if (data['capacity']-data["usage"]) < len(req):
                return False

        return True

    def run(self):
        for TClass in self._init_tactics:
            init_tactic = TClass(self)
            if not init_tactic.init_placement():
                return False
            self.save_best()
            if self.do_try():
                return True
        return False

    def do_try(self):
        effort = 10.0
        rare_reroute = 0
        for TClass in self._pick_tactics:
            tactic = TClass(self)
            no_improvement = 0
            while no_improvement <= tactic.max_no_improvement:
                if self.is_valid_embedding():
                    return True

                score = self.score()
                print(int(score), [self.placement[c] for c in self.constraints])
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

    def save_best(self, if_score=None):
        if self.best_found[0] is None or if_score < self.best_found[0]:
            self.best_found = (if_score, self.placement, self.chains)
            return True
        else:
            return False


    def is_valid_embedding(self):
        return all(data['usage'] <= data['capacity'] for _,data in self.arch.nodes(data=True))

    def scores(self):
        return {n:(exp(max(0, data['usage'] - data['capacity']))-1)/(math.e -1) for n, data in self.arch.nodes(data=True)}

    def score(self):
        return sum(self.scores().itervalues())

    def remove_tile(self, constr, clean_chains=True):
        old =  self.placement[constr]
        del self.placement[constr]
        if clean_chains:
            placement = self.tile_to_vars()
            for var_row, node in zip(constr.tile, old):
                for var in var_row:
                    if var not in self.chains or node not in self.chains[var]:
                        break # should be hit only when var has been deleted from the previous row
                    if var not in placement:
                        for deleted_node in self.chains[var]:
                            self.arch.nodes[deleted_node]['usage'] -= 1
                        del self.chains[var]
                        break
                    chaintree = nx.Graph(self.arch.subgraph(self.chains[var]))
                    leaf = node
                    while chaintree.degree(leaf) == 1 and leaf not in placement[var]:
                        newleaf = next(chaintree.neighbors(leaf))
                        chaintree.remove_node(leaf)
                        self.chains[var] = self.chains[var].difference({leaf})
                        self.arch.nodes[leaf]['usage'] -= 1
                        leaf = newleaf


    def tile_to_vars(self):
        placement = dict()
        for constraint, tile in self.placement.iteritems():
            for pnodes, anode in zip(constraint.tile, tile):
                for pnode in pnodes:
                    if pnode not in placement:
                        placement[pnode] = []
                    placement[pnode].append(anode)
        return placement

    def insert_tile(self, constraint, tile):
        # type: (Constraint, List[List[Any]]) -> None
        self._insert_tactic.insert_tile(constraint, tile)


