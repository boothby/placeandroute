import random

from collections import defaultdict
from itertools import combinations

from math import log
from typing import List, Any
from placeandroute.routing.bonnheuristic import MinMaxRouter
import networkx as nx


class Constraint(object):
    def __init__(self, vars):
        # type: (List[List[Any]]) -> None
        self.tile = vars


class TilePlacementHeuristic(object):
    def __init__(self, constraints, archGraph, choices):
        # type: (List[Constraint], nx.Graph, List[List[Any]]) -> None
        # list of constraints. each constraint has a set of variables and a weight. could use the factor graph
        self.constraints = constraints
        # target graph. each node is a set of edges in the original graph, each edge is a set of frontier vertices
        # somehow the possible choices are tagged
        # xxx: edges can be grouped if they are symmetrical, verify this
        self.arch = archGraph
        self.rounds = 1000
        self.counter = 0
        self.placement = dict()
        self.chains = {}
        self.choices = choices
        self.rip_tabu = []

    def constraint_fit(self, choice, constraint):
        for (node, req) in zip(choice, constraint.tile):
            data =self.arch.nodes[node]

            if (data['capacity']-data["usage"]) < len(req):
                return False

        return True

    def init_placement(self):
        choices = self.choices
        for constraint in self.constraints:
            possible_choices = [x for x in choices if self.constraint_fit(x, constraint)]
            if not possible_choices:
                return False
            position = random.choice(possible_choices)
            self.insert_tile(constraint, position)
        self.do_routing()
        return True

    def init_placement_bfs(self):
        cg = nx.Graph()
        vartoconst = defaultdict(list)
        for constraint in self.constraints:
            cg.add_node(constraint)
            for vset in constraint.tile:
                for v in vset:
                    vartoconst[v].append(constraint)

        for cset in vartoconst.itervalues():
            for v1,v2 in combinations(cset,2):
                cg.add_edge(v1,v2)

        centrality = nx.betweenness_centrality(cg)
        start = max(cg.nodes(), key=lambda x: centrality[x])
        starttile = self.choices[len(self.choices)/2]
        self.insert_tile(start, starttile)
        worst = None
        for _, constraint in nx.bfs_edges(cg, start):
            best = self.find_best_place(constraint)
            self.insert_tile(constraint, best)
            self.do_routing()
            worst = self.pick_worst()
            self.remove_tile(worst)
            best2 = self.find_best_place(worst)
            self.insert_tile(worst, best2)
        self.do_routing()

        return True


    def do_routing(self):
        placement = self.tile_to_vars()
        self.router = MinMaxRouter(self.arch, placement, epsilon=1)
        self.router.run(10)
        self.chains = self.router.result
        for node,data in self.arch.nodes(data=True):
            data["usage"] = 0
        for nodes in self.chains.itervalues():
            for node in nodes:
                self.arch.nodes[node]["usage"] += 1

    def run(self):
        if not self.init_placement():
            return False

        for round in xrange(self.rounds):
            if self.is_valid_embedding():
                return True
            print(self.score(), [self.placement[c] for c in self.constraints])
            remove_choice = self.pick_worst()
            self.remove_tile(remove_choice)
            best_place = self.find_best_place(remove_choice)
            self.insert_tile(remove_choice, best_place)
            self.do_routing()

        return False


    def is_valid_embedding(self):
        return all(data['usage'] <= data['capacity'] for _,data in self.arch.nodes(data=True))

    def score(self):
        return sum(max(0, data['usage'] - data['capacity']) for _, data in self.arch.nodes(data=True))

    def pick_worst_iter(self):
        ret = self.constraints[self.counter % len(self.constraints)]
        self.counter += 1
        return ret

    def pick_worst(self):
        costs = {node: max(0, data['usage'] - data['capacity']) for node, data in self.arch.nodes(data=True)}
        worstc = None
        worstcost = 0
        for constr in self.placement.iterkeys():
            if constr in self.rip_tabu: continue
            cost = sum(sum(sum(costs[node] for node in self.chains[var]) for var in var_row) for var_row in constr.tile)
            if worstc is None or cost > worstcost:
                worstc = constr
                worstcost = cost
        if worstc is None:
            worstc = random.choice(self.placement.keys())
        self.rip_tabu.append(worstc)
        if len(self.rip_tabu) > 10:
            self.rip_tabu.pop(0)
        return worstc


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


    def find_best_place(self, constraint):
        placement = self.tile_to_vars()
        router = MinMaxRouter(self.arch, placement, epsilon=log(2000))
        wgraph = router.wgraph
        #ephnodes = []
        scores = []
        for vset in constraint.tile:
            score = defaultdict(float)
            for v in vset:
                if v not in placement: continue
                ephnode = ("ephemeral",v)
                #ephnodes.append(ephnode)
                wgraph.add_edges_from([(ephnode,chain_node) for chain_node in placement[v]], weight=0)
                lengths = nx.single_source_dijkstra_path_length(wgraph, ephnode)
                for n, length in lengths.iteritems():
                    score[n] += length
                wgraph.remove_node(ephnode)
            scores.append(score)

        best_score = 0
        best_choice = None
        for choice in self.choices:
            score = scores[0][choice[0]] + scores[1][choice[1]]
            if best_choice is None or score < best_score:
                best_choice = choice
                best_score = score
        return best_choice


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
        self.placement[constraint] = tile


