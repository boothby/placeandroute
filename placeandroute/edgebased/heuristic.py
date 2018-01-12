import random as rand
from collections import defaultdict, Counter
import networkx as nx
from chimerautils.display import interactive_embeddingview
import os

from placeandroute.routing.bonnheuristic import MinMaxRouter


class EdgeHeuristic(object):
    def __init__(self, problemGraph, archGraph):
        # type: (nx.Graph, nx.Graph) -> None
        self.problemEdges = problemGraph.edges()  # list of pairs
        self.archEdges = archGraph.edges()  # list of pairs
        self.problemGraph = problemGraph  # nx.Graph
        self.archGraph = archGraph  # nx.Graph
        self.rounds = 1000
        self.exp = 2000.0
        self.assignment = None  # dict from problemEdges to archEdges
        self.tabu = list()
        self._tabu_size = max(problemGraph.degree(n) for n in problemGraph.nodes())

    def initial_assignment_random(self):
        self.assignment = dict(zip(self.problemEdges, rand.sample(self.archEdges, len(self.problemEdges))))
        self.find_routing(effort=10)

    def initial_assignment_greedy(self):
        self.assignment = dict()
        self.find_routing(effort=1)
        for pe in self.problemEdges:
            self._insert_edge(pe, effort=10)

    def _prefer_center(self, coords):
        def avg2d(vecs):
            vlen = len(vecs)
            retx, rety =  reduce(lambda (a, b), (c, d): (a + c, b + d), vecs)
            return retx/vlen, rety/vlen

        avgx, avgy = avg2d(coords)

        for n1, n2 in self.archEdges:
            edgex, edgey = avg2d([coords[n1], coords[n2]])
            self.archGraph[n1][n2]['distance_from_center'] = 0.001*((avgx - edgex)**2 + (avgy - edgey)**2)**0.5

    def _prefer_intracell(self):
        for n1, n2 in self.archEdges:
            self.archGraph[n1][n2]['distance_from_center'] += 10 if (n1//8) != (n2//8) else 0

    def initial_assignment_bfs(self, coords):
        self.assignment = dict()
        self.find_routing(effort=1)
        self._prefer_center(coords)
        self._prefer_intracell()
        centrality = Counter(nx.closeness_centrality(self.problemGraph))
        startnode = centrality.most_common(1)[0][0]
        ripped = None
        for edge in nx.bfs_edges(self.problemGraph, startnode):
            if ripped is not None:
                ripped = self.choose_ripped_edge()
                self._rip_edge(ripped, effort=10)
            bchoice = self.choose_best_edge(edge)
            self._insert_edge(edge, bchoice, effort=10)
            if ripped is not None:
                rchoice = self.choose_best_edge(ripped)
                self._insert_edge(ripped, rchoice, effort=10)
            else:
                ripped = 1

    def initial_assignment_local(self, coords):
        self.assignment = dict()
        self.find_routing(effort=1)
        self._prefer_center(coords)
        self._prefer_intracell()

        ripped = None
        for edge in local_order_edges(self.problemGraph):
            if ripped is not None:
                ripped = self.choose_ripped_edge()
                self._rip_edge(ripped, effort=10)
            bchoice = self.choose_best_edge(edge)
            self._insert_edge(edge, bchoice, effort=10)
            if ripped is not None:
                rchoice = self.choose_best_edge(ripped)
                self._insert_edge(ripped, rchoice, effort=10)
            else:
                ripped = 1



    def choose_ripped_edge(self):
        node_score = self._get_score()
        ret = None
        best_score = 0
        for k, v in self.assignment.iteritems():
            score = node_score[k[0]] + node_score[k[1]]
            if ret is None or  score > best_score and k not in self.tabu:
                ret = k
                best_score = score
        if len(self.tabu) == min(self._tabu_size, len(self.assignment)):
            self.tabu.pop(0)
        self.tabu.append(ret)
        return ret

    def choose_ripped_node(self):
        node_score = self._get_score()
        ret = max(node_score.items(), key=lambda (k,v): 0 if k in self.tabu else v)[1]

        if len(self.tabu) == min(self._tabu_size, len(self.assignment)):
            self.tabu.pop(0)
        self.tabu.append(ret)
        return ret

    def _get_score(self):
        node_score = dict()  # use this but weighted node usage not just count
        for n in self.problemGraph.nodes_iter():
            if not self.routing.is_present(n): continue
            node_score[n] = sum(self.archGraph[n1][n2]["weight"] for n1, n2 in self.routing.get_tree(n).edges_iter())
        return node_score

    def find_routing(self, effort):
        terminals = defaultdict(set)
        for pe, ae in self.assignment.iteritems():
            terminals[pe[0]].add(ae[0])
            terminals[pe[1]].add(ae[1])

        router = MinMaxRouter(self.archGraph, terminals)
        router.increase_weights(self.assignment.itervalues())
        router.run(effort)
        self.routing = router

        self._update_weights()

        if effort >= 10:
            os.environ["PATH"] += ":/home/svarotti/Drive/dwaveproj/projects/embeddingview"
            interactive_embeddingview(self.problemGraph, 16, False)

    def _update_weights(self):
        router = self.routing

        for n1 in self.archGraph.nodes_iter():
            self.archGraph.node[n1]["mappedto"] = set()

        for node in self.problemGraph.nodes_iter():
            if router.is_present(node):
                nn = router.get_nodes(node)
                for nnn in nn:
                    self.archGraph.node[nnn]["mappedto"].add(node)
            else:
                nn = list()
            self.problemGraph.node[node]["mapto"] = nn

        getsize = lambda n: len(self.archGraph.node[n]["mappedto"])
        for n1, n2 in self.archGraph.edges_iter():
            self.archGraph[n1][n2]["weight"] = max([self.exp**getsize(n1), self.exp**getsize(n2)])


    def choose_best_edge(self, probEdge):
        self._bestEdge = None
        self._bestDist = None
        self._probEdge = probEdge
        self._distances = [
        self._prepare_distances(probEdge[0]),
        self._prepare_distances(probEdge[1])]


        for n1, n2 in self.archEdges:
            self._update_optimum(n1, n2)
            self._update_optimum(n2, n1) #edge choice can be flipped

        return self._bestEdge

    def _prepare_distances(self, pnode):
        ephemeral = ("ephemeral node")

        if not self.routing.is_present(pnode):
            return {tonode: 0 *self.exp ** len(self.archGraph.node[tonode]["mappedto"]) for tonode in self.archGraph.nodes_iter()}

        nodes = self.routing.get_tree(pnode).nodes()
        for node in nodes:
            self.archGraph.add_edge(ephemeral, node, weight=0)
        ret = nx.single_source_dijkstra_path_length(self.archGraph, ephemeral, weight=None)
        self.archGraph.remove_node(ephemeral)
        return ret

    def _node_distance(self, p, tonode):
        return self._distances[self._probEdge.index(p)][tonode]


    def _update_optimum(self, n1, n2):
        p1, p2 = self._probEdge
        dist = self._node_distance(p1, n1) + self._node_distance(p2, n2) \
               + self.archGraph[n1][n2]['distance_from_center'] \
               + self.exp ** max(len(self.archGraph.node[n1]["mappedto"].difference({p1})),
                len(self.archGraph.node[n2]["mappedto"].difference({p2})))
        if self._bestEdge is None or dist < self._bestDist:
            self._bestEdge = (n1, n2)
            self._bestDist = dist

    def run(self):
        if self.assignment is None:
            self.initial_assignment_random()
            for round in xrange(self.rounds):
                print sum(self._get_score().itervalues()) # bad count
                if all(len(self.archGraph.node[n]["mappedto"]) < 2 for n in self.archGraph.nodes_iter()):
                    break
                self.reroute_node()

    def reroute_edge(self):
        choice = self.choose_ripped_edge()
        self._rip_edge(choice,effort=10)
        dest = self.choose_best_edge(choice)
        self._insert_edge(choice, dest, effort=10)

    def reroute_node(self):
        chnode = self.choose_ripped_node()
        redges = [e for e in self.problemEdges if chnode in e]
        for e in redges[:-1]:
            self._rip_edge(e, effort=0)
        self._rip_edge(redges[-1],effort=10)
        rand.shuffle(redges)
        for e in redges:
            choice = self.choose_best_edge(e)
            self._insert_edge(e, choice, effort=10)



    def _edge_usage(self):
        count = Counter()
        for tree in self.routing.result.values():
            count.update(map(frozenset,tree.edges()))
        for aed in self.assignment.values():
            count[frozenset(aed)] += 1
        return count

    def _node_cost(self):
        return sum(self.exp **len(d["mappedto"]) for n,d in self.archGraph.nodes_iter(data=True))

    def _insert_edge(self, choice, dest, effort):
        self.assignment[choice] = dest
        if effort: self.find_routing(effort)

    def _rip_edge(self, choice, effort):
        del self.assignment[choice]
        if effort: self.find_routing(effort)


def local_order_edges(p):
    n = {max(p.nodes(), key=p.degree)}
    edge_ordered = []
    while True:
        nn = set()
        for x in n:
            nn.update(p.neighbors(x))
        nn.difference_update(n)
        if not len(nn): break
        nn = max(nn, key=p.degree)
        edge_ordered.extend((nn, o) for o in p.neighbors(nn) if o in n)
        n.add(nn)
    return edge_ordered