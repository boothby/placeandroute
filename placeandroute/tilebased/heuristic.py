import random

from collections import defaultdict
from itertools import combinations

from typing import List, Any
from placeandroute.routing.bonnheuristic import MinMaxRouter
import networkx as nx


class Constraint(object):
    def __init__(self, vars):
        # type: (List[List[Any]]) -> None
        self.tile = vars


def chimeratiles(w,h):
    ret = nx.Graph()
    i = 0
    grid = []
    choices = []
    for c in range(w):
        row = []
        for r in range(h):
            n1, n2 = i, i +1
            i += 2
            ret.add_nodes_from([n1,n2], capacity=4, usage=0)
            ret.add_edge(n1,n2, capacity=16)
            if row:
                ret.add_edge(n1, row[-1][0], capacity=4)
            if grid:
                ret.add_edge(n2,grid[-1][len(row)][1],capacity=4)
            row.append((n1,n2))
            choices.extend([(n1,n2), (n2,n1)])
        grid.append(row)
    return ret, choices

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

    def constraint_fit(self, choice, constraint):
        for (node, req) in zip(choice, constraint.tile):
            data =self.arch.nodes[node]

            if (data['capacity']-data["usage"]) < len(req):
                return False

        return True

    def init_placement_random(self):
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
        for _, constraint in nx.bfs_edges(cg, start):
            best = self.find_best_place(constraint)
            self.insert_tile(constraint, best)
            #worst = self.pick_worst()
            #self.remove_tile(worst)
            #best2 = self.find_best_place(worst)
            #self.insert_tile(worst, best2)

        return True


    def do_routing(self):
        placement = self.tile_to_vars()
        self.router = MinMaxRouter(self.arch, placement, capacity=4, epsilon=0.001)
        self.router.run(10)
        self.chains = self.router.result
        for node,data in self.arch.nodes(data=True):
            data["usage"] = 0
        for nodes in self.chains.itervalues():
            for node in nodes:
                self.arch.nodes[node]["usage"] += 1

    def run(self):
        if not self.init_placement_random():
            return False

        for round in xrange(self.rounds):
            if self.is_valid_embedding():
                return True
            print(self.score())
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

    def pick_worst(self):
        ret = self.constraints[self.counter % len(self.constraints)]
        self.counter += 1
        return ret


    def remove_tile(self, constr, clean_chains=True):
        del self.placement[constr]
        if clean_chains:
            # todo: only remove dead branches
            self.do_routing()

    def find_best_place(self, constraint):
        placement = self.tile_to_vars()
        router = MinMaxRouter(self.arch, placement, capacity=4, epsilon=0.001)
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
        self.do_routing()
