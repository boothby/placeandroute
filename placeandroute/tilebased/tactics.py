import random
from itertools import combinations
from math import exp, log

import networkx as nx
from collections import defaultdict

from placeandroute.routing.bonnheuristic import MinMaxRouter
#from .heuristic import TilePlacementHeuristic, Constraint

class Tactic(object):
    """Generic tactic object"""
    def __init__(self, parent):
        # type: (TilePlacementHeuristic) -> None
        self.parent = parent

    def init_placement(self):
        # type: () -> bool
        raise NotImplementedError

    def pick_worst(self):
        # type: () -> Constraint
        raise NotImplementedError


class RandomInitTactic(Tactic):
    def init_placement(self):
        p = self.parent
        choices = p.choices
        for constraint in p.constraints:
            possible_choices = [x for x in choices if p.constraint_fit(x, constraint)]
            if not possible_choices:
                return False
            position = random.choice(possible_choices)
            p.insert_tile(constraint, position)
        RerouteInsertTactic(p).do_routing()
        return True


class BFSInitTactic(Tactic):
    def init_placement(self):
        p = self.parent
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

        pick_t = CostlyTilePickTactic(p)
        centrality = nx.betweenness_centrality(cg)
        start = max(cg.nodes(), key=lambda x: centrality[x])
        arch_centrality = nx.betweenness_centrality(p.arch)
        starttile = max(p.choices, key=lambda (x,y): arch_centrality[x] + arch_centrality[y])
        p.insert_tile(start, starttile)
        #best_place = RoughFindPlaceTactic(p)
        best_place = ChainsFindPlaceTactic(p)


        for _, constraint in nx.bfs_edges(cg, start):
            best = best_place.find_best_place(constraint)
            p.insert_tile(constraint, best)
            worst = pick_t.pick_worst()
            p.remove_tile(worst)
            best2 = best_place.find_best_place(worst)
            p.insert_tile(worst, best2)

        return True

class RipRerouteTactic(Tactic):
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
    max_no_improvement = 400
    def __init__(self, p):
        Tactic.__init__(self, p)
        self.rip_tabu = []

    def pick_worst(self):
        p = self.parent
        costs = {node: exp(max(0, data['usage'] - data['capacity']))-1 for node, data in p.arch.nodes(data=True)}
        worstc = None
        worstcost = 0
        for constr in p.placement.iterkeys():
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


class RoughFindPlaceTactic(Tactic):
    def find_best_place(self, constraint):
        parent = self.parent
        placement = parent.tile_to_vars()
        epsilon = log(sum(d["capacity"] for _, d in parent.arch.nodes(data=True)))

        router = MinMaxRouter(parent.arch, placement, epsilon=epsilon)
        wgraph = router.weights_graph

        scores = []
        for vset in constraint.tile:
            score = defaultdict(float)
            for v in vset:
                if v not in placement: continue
                for chain_node in placement[v]:
                    data = wgraph.nodes[chain_node]
                    score[chain_node] = exp(epsilon*max(0, data['usage']-data['capacity']))-1

                ephnode = ("ephemeral",v)
                wgraph.add_edges_from([(ephnode,chain_node) for chain_node in placement[v]], weight=0)
                lengths = nx.single_source_dijkstra_path_length(wgraph, ephnode)
                for n, length in lengths.iteritems():
                    score[n] += length
                wgraph.remove_node(ephnode)
            scores.append(score)

        best_score = 0
        best_choice = None
        for choice in parent.choices:
            score = scores[0][choice[0]] + scores[1][choice[1]]
            if best_choice is None or score < best_score:
                best_choice = choice
                best_score = score
        return best_choice


class ChainsFindPlaceTactic(Tactic):
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
            unrelated = [mapsto[target].difference(varrow) for varrow, target in zip(constraint.tile, choice)]
            unrel_score = sum(exp(scaling_factor * max(0, len(unrel_nodes)- wgraph.nodes[target]["capacity"]))-1
                              for unrel_nodes, target in zip(unrelated, choice))
            score = scores[0][choice[0]] + scores[1][choice[1]] + unrel_score
            if best_choice is None or score < best_score:
                best_choice = choice
                best_score = score
        return best_choice


class InsertTactic(Tactic):
    def insert_tile(self, constraint, tile):
        # type: (Constraint, List[List[Any]]) -> None
        p = self.parent
        p.placement[constraint] = tile
        self.do_routing()

    def do_routing(self):
        pass


class RerouteInsertTactic(InsertTactic):
    def do_routing(self, effort=100):
        print "Rerouting..."
        p = self.parent
        placement = p.tile_to_vars()
        router = MinMaxRouter(p.arch, placement, epsilon=1.0/effort)
        router.run(effort)
        p.chains = router.result
        for node,data in p.arch.nodes(data=True):
            data["usage"] = 0
        for nodes in p.chains.itervalues():
            for node in nodes:
                p.arch.nodes[node]["usage"] += 1


class IncrementalRerouteInsertTactic(InsertTactic):
    def do_routing(self, effort=100):
        p = self.parent
        placement = p.tile_to_vars()
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


class SimpleInsertTactic(InsertTactic):
    def insert_tile(self, constraint, tile):
        # type: (Constraint, List[List[Any]]) -> None
        p = self.parent
        p.placement[constraint] = tile
        #add shortest-path chains
        chains = p.chains
        wgraph = nx.DiGraph(p.arch)
        def set_weights(nodeset):
            for node in nodeset:
                ndat = wgraph.nodes[node]
                usage = max(0, ndat["usage"] - ndat["capacity"])
                for _, _, data in wgraph.in_edges(node, data=True):
                    data["weight"] = exp(log(2000) * usage)

        set_weights(wgraph.nodes)

        for var_row,choice in zip(constraint.tile, tile):
            for var in var_row:
                if var not in chains:
                    chains[var] = frozenset({choice})
                    wgraph.nodes[choice]["usage"] += 1
                    set_weights([choice])
                    continue
                _, path = nx.multi_source_dijkstra(wgraph, chains[var], choice)
                path = set(path)
                path.difference_update(chains[var])
                for node in path:
                    wgraph.nodes[node]["usage"] += 1
                set_weights(path)
                chains[var]= chains[var].union(frozenset(path))

        for node, data in p.arch.nodes(data=True):
            data["usage"] = 0
        for nodes in p.chains.itervalues():
            for node in nodes:
                p.arch.nodes[node]["usage"] += 1
        for node in wgraph.nodes:
            assert wgraph.nodes[node]["usage"] == p.arch.nodes[node]["usage"]