import random
from itertools import combinations
from math import exp, log

import networkx as nx
from collections import defaultdict

from placeandroute.routing.bonnheuristic import MinMaxRouter
#from .heuristic import TilePlacementHeuristic, Constraint

class Tactic(object):
    """Generic tactic object. Will hold the TileHeuristic object in the parent attribute"""
    def __init__(self, parent):
        # type: (TilePlacementHeuristic) -> None
        self.parent = parent

    def init_placement(self):
        # type: () -> bool
        #todo: abstract InitTactic, maybe
        raise NotImplementedError



class RandomInitTactic(Tactic):
    """Initialize placement by random assignment"""

    def constraint_fit(self, choice, constraint):
        for (node, req) in zip(choice, constraint.tile):
            data =self.parent.arch.nodes[node]

            if (data['capacity']-data["usage"]) < len(req):
                return False

        return True
    def init_placement(self):
        p = self.parent
        choices = p.choices
        for constraint in p.constraints:
            possible_choices = [x for x in choices if self.constraint_fit(x, constraint)]
            if not possible_choices:
                return False
            position = random.choice(possible_choices)
            p.insert_tile(constraint, position)
        RerouteInsertTactic(p).do_routing()
        return True


class BFSInitTactic(Tactic):
    """Initialize placement by breadth-first-search. Identifies a "central" constraint, traverses the constraint graph
    using BFS, placing and routing each constraint. After inserting a new constraint, rip&reroute one constraint to
    correct overusage early"""
    def init_placement(self):
        p = self.parent

        #build the constraint graph
        cg = nx.Graph()
        vartoconst = defaultdict(list)
        for constraint in p.constraints:
            cg.add_node(constraint)
            for vset in constraint.tile:
                for v in vset:
                    vartoconst[v].append(constraint)

        for cset in vartoconst.itervalues():
            for v1,v2 in combinations(cset,2):
                cg.add_edge(v1,v2)

        pick_t = CostlyTilePickTactic(p) # pick tactic

        # "central" constraint in problem graph
        centrality = nx.betweenness_centrality(cg)
        start = max(cg.nodes(), key=lambda x: centrality[x])

        # "central" tile in arch graph
        arch_centrality = nx.betweenness_centrality(p.arch)
        starttile = max(p.choices, key=lambda (x,y): arch_centrality[x] + arch_centrality[y])
        p.insert_tile(start, starttile)
        #best_place = RoughFindPlaceTactic(p)
        best_place = ChainsFindPlaceTactic(p)


        for _, constraint in nx.bfs_edges(cg, start):
            # insert new constraint in best place
            best = best_place.find_best_place(constraint)
            p.insert_tile(constraint, best)
            # rip and reroute the worst constraint
            worst = pick_t.pick_worst()
            p.remove_tile(worst)
            best2 = best_place.find_best_place(worst)
            p.insert_tile(worst, best2)

        return True

class RipRerouteTactic(Tactic):
    # WORK IN PROGRESS: unused yet
    def __init__(self, p, removeTactic, findTactic, insertTactic):
        Tactic.__init__(self, p)
        self._remove = removeTactic(p)
        self._find = findTactic(p)
        self._insert = insertTactic(p)

    def improve(self):
        remove_choice = self.pick_worst()
        self.remove_tile(remove_choice)
        # print self.score()
        best_place = self.find_best_place(remove_choice)
        self.insert_tile(remove_choice, best_place)



class IterPickTactic(Tactic):
    """Choose a constraint to reroute using round-robin"""
    def __init__(self, p):
        Tactic.__init__(self, p)
        self.counter = 0
        self.max_no_improvement = len(p.constraints) #* 10

    def pick_worst(self):
        p = self.parent
        ret = p.constraints[self.counter % len(p.constraints)]
        self.counter += 1
        return ret


class CostlyTilePickTactic(Tactic):
    """Reroute choice based on cost (~ e**overusage). Uses a tabu list to avoid rerouting the same few constraints

     todo: calculate cost in only one place
    """
    max_no_improvement = 400
    def __init__(self, p):
        Tactic.__init__(self, p)
        self.rip_tabu = []

    def pick_worst(self):
        p = self.parent
        costs = {node: exp(max(0, data['usage'] - data['capacity']))-1 for node, data in p.arch.nodes(data=True)}
        worstc = None
        worstcost = 0
        for constr in p.constraint_placement.iterkeys():
            if constr in self.rip_tabu: continue
            cost = sum(sum(sum(costs[node] for node in p.chains[var]) for var in var_row) for var_row in constr.tile)
            if worstc is None or cost > worstcost:
                worstc = constr
                worstcost = cost
        if worstc is None:
            worstc = random.choice(p.placement.keys())
        self.rip_tabu.append(worstc)
        if len(self.rip_tabu) > 10:
            self.rip_tabu.pop(0)
        return worstc


class ChainsFindPlaceTactic(Tactic):
    """Choose the best tile for a constraint based on shortest path from already-present chains"""
    def find_best_place(self, constraint):
        parent = self.parent
        placement = parent.chains
        scaling_factor = log(2000)
        wgraph = nx.DiGraph(parent.arch)

        def set_weights(nodeset):
            for node in nodeset:
                ndat = wgraph.nodes[node]
                usage = max(float(ndat["usage"] )/ ndat["capacity"], ndat["usage"] - ndat["capacity"])
                for _, _, data in wgraph.in_edges(node, data=True):
                    data["weight"] = exp(scaling_factor * usage)

        set_weights(wgraph.nodes)

        #For each arch node calc a score
        scores = []

        for vset in constraint.tile:
            score = defaultdict(float)
            for v in vset:
                if v not in placement: continue
                ephnode = ("ephemeral",v)
                wgraph.add_edges_from([(ephnode,chain_node) for chain_node in placement[v]], weight=0)
                lengths = nx.single_source_dijkstra_path_length(wgraph, ephnode)
                for n, length in lengths.iteritems():
                    score[n] += length
                wgraph.remove_node(ephnode)
            scores.append(score)

        best_score = 0
        best_choice = None
        mapsto = defaultdict(set)
        for k, vs in placement.iteritems():
            for v in vs:
                mapsto[v].add(k)

        for choice in parent.choices:
            # variables mapped to this choice that are unrelated to this constraints
            unrelated = [mapsto[target].difference(varrow) for varrow, target in zip(constraint.tile, choice)]
            unrel_score = sum(exp(scaling_factor * max(0, len(unrel_nodes)- wgraph.nodes[target]["capacity"]))-1
                              for unrel_nodes, target in zip(unrelated, choice))

            score = scores[0][choice[0]] + scores[1][choice[1]] + unrel_score

            if best_choice is None or score < best_score:
                best_choice = choice
                best_score = score
        return best_choice


class InsertTactic(Tactic):
    # generic re-routing tactic: todo: will be factored away
    def insert_tile(self, constraint, tile):
        # type: (Constraint, List[List[Any]]) -> None
        p = self.parent
        p.constraint_placement[constraint] = tile
        self.do_routing()

    def do_routing(self):
        pass


class RerouteInsertTactic(InsertTactic):
    """Throw away the current chain and reroute everything. Uses bonnroute."""
    def do_routing(self, effort=100):
        print "Rerouting..."
        p = self.parent
        placement = p.var_placement()
        router = MinMaxRouter(p.arch, placement, epsilon=1.0/effort)
        router.run(effort)
        p.chains = router.result
        for node,data in p.arch.nodes(data=True):
            data["usage"] = 0
        for nodes in p.chains.itervalues():
            for node in nodes:
                p.arch.nodes[node]["usage"] += 1


class IncrementalRerouteInsertTactic(InsertTactic):
    """Perform routing keeping the currently available chains. Uses bonnroute, but essentially
    forces the use of old chains"""
    def do_routing(self, effort=100):
        p = self.parent
        placement = p.var_placement()
        for k, vs in p.chains.iteritems():
            placement[k] = set(placement[k]).union(vs)
        router = MinMaxRouter(p.arch, placement, epsilon=1.0/effort)
        router.run(effort)
        p.chains = router.result
        for node,data in p.arch.nodes(data=True):
            data["usage"] = 0
        for nodes in p.chains.itervalues():
            for node in nodes:
                p.arch.nodes[node]["usage"] += 1
