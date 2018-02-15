import random

from collections import defaultdict
from itertools import combinations

from math import log, exp
from typing import List, Any, overload
from placeandroute.routing.bonnheuristic import MinMaxRouter
import networkx as nx


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
        p.do_routing()
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
        best_place = RoughFindPlaceTactic(p)

        for _, constraint in nx.bfs_edges(cg, start):
            best = best_place.find_best_place(constraint)
            p.insert_tile(constraint, best)
            worst = pick_t.pick_worst()
            p.remove_tile(worst)
            best2 = best_place.find_best_place(worst)
            p.insert_tile(worst, best2)

        return True

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
    max_no_improvement = 1000
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
        wgraph = router.wgraph

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
        wgraph = nx.DiGraph(parent.arch)

        def set_weights(nodeset):
            for node in nodeset:
                ndat = wgraph.nodes[node]
                usage = max(float(ndat["usage"] )/ ndat["capacity"], ndat["usage"] - ndat["capacity"])
                for _, _, data in wgraph.in_edges(node, data=True):
                    data["weight"] = exp(log(2000) * usage)

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
        for choice in parent.choices:
            score = scores[0][choice[0]] + scores[1][choice[1]]
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
        self._init_tactics = [BFSInitTactic, RandomInitTactic]*3+[RandomInitTactic]*4
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
                print(score, [self.placement[c] for c in self.constraints])
                if self.save_best(if_score=score):
                    no_improvement = 0
                else:
                    no_improvement += 1
                    rare_reroute += 1
                if rare_reroute > 20:
                        rare_reroute = 0
                        effort += 5.0
                        RerouteInsertTactic(self).do_routing(int(effort))
                    #effort += 0.1
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
        return {n:exp(max(0, data['usage'] - data['capacity']))-1 for n, data in self.arch.nodes(data=True)}

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


