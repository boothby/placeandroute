import logging
import random
from collections import defaultdict
from itertools import combinations
from math import log

import networkx as nx
from six import itervalues, iteritems
from typing import List, Any

from placeandroute.routing.bonnheuristic import MinMaxRouter, bounded_exp, fast_steiner_tree
try:
    import placeandroutecpp
    MinMaxRouter = placeandroutecpp.MinMaxRouter
    logging.info("loaded placeandroutecpp router")
except ModuleNotFoundError as e:
    logging.info("no cpp version, using pure python: %r", e)
# from .heuristic import TilePlacementHeuristic, Constraint

class TacticFactory(object):
    def __init__(self, tactic, **params):
        self.tactic = tactic
        self.params = params

    def create(self, placement):
        return self.tactic(placement, **self.params)


class Tactic(object):
    """Generic tactic object. Will hold the TileHeuristic object in the parent attribute"""

    @classmethod
    def default(cls):
        return TacticFactory(cls)


    def __init__(self, placement):
        self._placement = placement

    def run(self):
        pass

    def run_tactic(self):
        logging.info("Running %s", self)
        self.run()

    def __str__(self):
        return "[unknown Tactic]"


class RepeatTactic(Tactic):
    """Repeats a tactic until there is no improvement for max turns"""
    def __init__(self, placement, subTactic, max_no_improvement):
        Tactic.__init__(self, placement)
        self._sub_tactic = subTactic.create(placement)
        self._max_no_improvement = max_no_improvement

    @classmethod
    def default(cls):
        raise NotImplementedError

    @classmethod
    def with_(cls, subfact, maxnoimp):
        return TacticFactory(cls, subTactic=subfact, max_no_improvement=maxnoimp)

    def __str__(self):
        return "[repeated {!s}, max stall {}]".format(self._sub_tactic, self._max_no_improvement)

    def clear_best(self):
        self.best_found = (None, None, None)

    def run(self):
        no_improvement = 0
        tactic = self._sub_tactic
        self.clear_best()
        self.save_best()
        while no_improvement < self._max_no_improvement:
            tactic.run()
            if not self.save_best():
                no_improvement += 1
            else:
                no_improvement = 0
        (if_score, self._placement.constraint_placement, self._placement.chains) = self.best_found
        self._placement.fix_usage()

    def save_best(self):
        # type: () -> bool
        """Check if the current result has been the best so far"""
        if_score = self._placement.score()
        if self.best_found[0] is None or if_score < self.best_found[0]:
            self.best_found = (if_score, self._placement.constraint_placement.copy(), self._placement.chains.copy())
            return True
        else:
            return False


class RandomInitTactic(Tactic):
    """Initialize placement by random assignment"""
    def __str__(self):
        return "[random initialization]"

    def constraint_fit(self, choice, tile, constraint):
        for (node, req) in zip(choice, tile):
            data = self._placement.arch.nodes[node]

            if (data['capacity'] - data["usage"]) < len(req):
                return False

        return True

    def run(self):
        p = self._placement
        choices = p.choices
        # insert_tactic = InsertTactic(p, IncrementalRerouteTactic)
        for constraint in p.constraints:
            tile = constraint.random_placement
            possible_choices = [x for x in choices if self.constraint_fit(x, tile, constraint)]
            if not possible_choices:
                possible_choices = choices
            position = random.choice(possible_choices)
            p.constraint_placement[constraint] = (tile, position)
        final_reroute = RerouteTactic(p)
        final_reroute.do_routing()


class BFSInitTactic(Tactic):
    """Initialize placement by breadth-first-search. Identifies a "central" constraint, traverses the constraint graph
    using BFS, placing and routing each constraint. After inserting a new constraint, rip&reroute one constraint to
    correct overusage early"""

    def __str__(self):
        return "[BFS heuristic initialization]"

    def run(self):
        p = self._placement
        # build the constraint graph
        cg = nx.Graph()
        vartoconst = defaultdict(list)
        for constraint in p.constraints:
            tile = constraint.random_placement
            cg.add_node(constraint)
            for vset in tile:
                for v in vset:
                    vartoconst[v].append(constraint)

        for cset in itervalues(vartoconst):
            for v1, v2 in combinations(cset, 2):
                cg.add_edge(v1, v2)

        pick_t = CostPickTactic(p)  # pick tactic
        insert_tactic = InsertTactic.quick().create(p)
        remove_tactic = RipTactic(p)
        best_place = ChainsFindTactic(p)


        # "central" constraint in problem graph
        centrality = nx.betweenness_centrality(cg)
        start = max(cg.nodes(), key=lambda x: centrality[x])

        # "central" tile in arch graph
        arch_centrality = nx.betweenness_centrality(p.arch)
        starttile = max(p.choices, key=lambda xy: arch_centrality[xy[0]] + arch_centrality[xy[1]])
        insert_tactic.insert_tile(start, start.random_placement, starttile)

        for _, constraint in nx.bfs_edges(cg, start):
            # insert new constraint in best place
            bestpl, best = best_place.find_best_place(constraint)
            insert_tactic.insert_tile(constraint, bestpl, best)
            # rip and reroute the worst constraint
            worst = pick_t.pick_worst()
            remove_tactic.remove_tile(worst)
            bestpl2, best2 = best_place.find_best_place(worst)
            insert_tactic.insert_tile(worst, bestpl2, best2)

        return True


class RipTactic(Tactic):
    """Rips a constraints from the placements and trims the chains"""
    def __str__(self):
        return "[trim chains]"

    def remove_tile(self, constr, clean_chains=True):
        # remove a constraint from the current embedding.
        place_obj = self._placement
        oldtile, old = self._placement.constraint_placement[constr]
        del self._placement.constraint_placement[constr]
        if clean_chains:
            var_placement = place_obj.var_placement()
            for var_row, node in zip(oldtile, old):
                for var in var_row:
                    if var not in place_obj.chains or node not in place_obj.chains[var]:
                        continue  # should be hit only when var has been deleted from the previous row

                    if var not in var_placement:
                        # the deleted tile was the only one containing this variables, clear the whole chain
                        for deleted_node in place_obj.chains[var]:
                            place_obj.arch.nodes[deleted_node]['usage'] -= 1
                        del place_obj.chains[var]
                        continue

                    chaintree = nx.Graph(place_obj.arch.subgraph(place_obj.chains[var]))
                    leaf = node
                    # remove only unused qubits from the chain (leaf in the tree that do not belong to other constraints
                    while chaintree.degree[leaf] == 1 and leaf not in var_placement[var]:
                        newleaf = next(chaintree.neighbors(leaf))
                        chaintree.remove_node(leaf)
                        place_obj.chains[var] = place_obj.chains[var].difference({leaf})
                        place_obj.arch.nodes[leaf]['usage'] -= 1
                        leaf = newleaf



class IterPickTactic(Tactic):
    """Choose a constraint to reroute using round-robin"""
    def __str__(self):
        return "[round-robin]"

    def __init__(self, placement):
        Tactic.__init__(self, placement)
        self.counter = 0

    def pick_worst(self):
        p = self._placement
        ret = p.constraints[self.counter % len(p.constraints)]
        self.counter += 1
        return ret


class CostPickTactic(Tactic):
    """Reroute choice based on cost (~ e**overusage). Uses a tabu list to avoid rerouting the same few constraints

     todo: calculate cost in only one place
    """

    def __init__(self, placement, tabu_size=10):
        Tactic.__init__(self, placement)
        self.tabu_size = tabu_size
        self.rip_tabu = []

    def __str__(self):
        return "[cost estimation]"

    def pick_worst(self):
        p = self._placement
        costs = p.scores()
        worstc = None
        worstcost = 0
        for constr, (tile_placement, _) in p.constraint_placement.items():
            if constr in self.rip_tabu: continue
            cost = sum(sum(sum(costs[node] for node in p.chains[var]) for var in var_row) for var_row in tile_placement)
            if worstc is None or cost > worstcost:
                worstc = constr
                worstcost = cost
        if worstc is None:
            worstc = random.choice(list(p.constraint_placement.keys()))
        self.rip_tabu.append(worstc)
        if len(self.rip_tabu) > self.tabu_size:
            self.rip_tabu.pop(0)
        return worstc


class ChainsFindTactic(Tactic):
    """Choose the best tile for a constraint based on shortest path from already-present chains"""

    def __init__(self, placement):
        Tactic.__init__(self,placement)
        self.scaling_factor = placement.coeff

    def __str__(self):
        return "[cost estimation]"

    def find_best_place(self, constraint):
        parent = self._placement
        placement = parent.chains
        scaling_factor = self.scaling_factor
        wgraph = nx.DiGraph(parent.arch)

        def set_weights(nodeset):
            for node in nodeset:
                ndat = wgraph.nodes[node]
                usage = max(float(ndat["usage"]) / ndat["capacity"], ndat["usage"] - ndat["capacity"])
                for _, _, data in wgraph.in_edges(node, data=True):
                    data["weight"] = bounded_exp(scaling_factor * usage)

        set_weights(wgraph.nodes)

        best_score = 0
        best_choice = None
        for tile_placement in constraint.placements:
            # For each arch node calc a score
            scores = []
            for vset in tile_placement:
                score = defaultdict(float)
                for v in vset:
                    if v not in placement: continue
                    ephnode = ("ephemeral", v)
                    wgraph.add_edges_from([(ephnode, chain_node) for chain_node in placement[v]], weight=0)
                    lengths = nx.single_source_dijkstra_path_length(wgraph, ephnode)
                    for n, length in iteritems(lengths):
                        score[n] += length
                    wgraph.remove_node(ephnode)
                scores.append(score)


            mapsto = defaultdict(set)
            for k, vs in iteritems(placement):
                for v in vs:
                    mapsto[v].add(k)

            for choice in parent.choices:
                # variables mapped to this choice that are unrelated to this constraints
                unrelated = [mapsto[target].difference(varrow) for varrow, target in zip(tile_placement, choice)]
                unrel_score = sum(
                    bounded_exp(scaling_factor * max(0, len(unrel_nodes) - wgraph.nodes[target]["capacity"])) - 1
                                                                for unrel_nodes, target in zip(unrelated, choice))

                score = scores[0][choice[0]] + scores[1][choice[1]] + unrel_score

                if best_choice is None or score < best_score:
                    best_choice = (tile_placement, choice)
                    best_score = score
        return best_choice


class RerouteTactic(Tactic):
    """Throws away the current chain and reroute everything. Uses bonnroute."""

    def __init__(self, placement, effort=100, steiner_func=fast_steiner_tree, astar_heuristic=False):
        Tactic.__init__(self, placement)
        self.effort = effort
        self.steiner_func = steiner_func
        self._astar = astar_heuristic
        self.router = MinMaxRouter(placement.arch, steiner_func=steiner_func, astar_heuristic=astar_heuristic)


    def __str__(self):
        return "[bonnroute on everything]"

    def do_routing(self):
        p = self._placement
        effort = self.effort
        placement = p.var_placement()
        p.chains = self.router.run(placement, epsilon=float(p.coeff) / effort, effort=effort)
        p.fix_usage()

    def run(self):
        self.do_routing()


class IncrementalRerouteTactic(RerouteTactic):
    """Perform routing keeping the currently available chains. Uses bonnroute, but essentially
    forces the use of old chains"""
    def __str__(self):
        return "[incremental bonnroute]"

    def do_routing(self):
        p = self._placement
        effort = self.effort
        placement = p.var_placement()
        for k, vs in iteritems(p.chains):
            placement[k] = set(placement[k]).union(vs)
        p.chains = self.router.run(placement, float(p.coeff) / effort, effort)
        p.fix_usage()


class InsertTactic(Tactic):
    """Insert a constraint somewhere and reroute"""

    @classmethod
    def quick(cls):
        return TacticFactory(cls, routingTactic=IncrementalRerouteTactic.default())

    @classmethod
    def slow(cls):
        return TacticFactory(cls, routingTactic=RerouteTactic.default())

    def __str__(self):
        return "[insert, then {}]".format(self._routing_tactic)

    def __init__(self, placement, routingTactic):
        Tactic.__init__(self, placement)
        self._routing_tactic = routingTactic.create(placement)

    def insert_tile(self, constraint, chosen_placement, tile):
        # type: (Constraint, List[List[Any]],List[List[Any]]) -> None
        p = self._placement
        p.constraint_placement[constraint] = (chosen_placement,tile)
        self._routing_tactic.do_routing()


class RipRerouteTactic(Tactic):
    """Main rip&reroute class"""

    @classmethod
    def repeated(cls):
        return RepeatTactic.with_(subfact=cls.default(), maxnoimp=50)

    def __init__(self, placement, removeTactic=RipTactic.default(), findTactic=ChainsFindTactic.default(), insertTactic=InsertTactic.quick(),
                 pickTactic=CostPickTactic.default()):
        Tactic.__init__(self, placement)
        self._remove = removeTactic.create(placement)
        self._find = findTactic.create(placement)
        self._insert = insertTactic.create(placement)
        self._pick = pickTactic.create(placement)


    def __str__(self):
        return "[rip&reroute, remove: {!s} find: {!s} insert: {!s} pick: {!s}]".format(self._remove, self._find,
                                                                                       self._insert, self._pick)

    def run(self):
        remove_choice = self.pick_worst()
        self.remove_tile(remove_choice)
        best_tile, best_place = self.find_best_place(remove_choice)
        self.insert_tile(remove_choice, best_tile, best_place)

    def pick_worst(self):
        return self._pick.pick_worst()

    def remove_tile(self, choice):
        self._remove.remove_tile(choice)

    def find_best_place(self, choice):
        return self._find.find_best_place(choice)

    def insert_tile(self, constr, tile, place):
        self._insert.insert_tile(constr, tile, place)
