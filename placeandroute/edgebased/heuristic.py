import random as rand
from collections import defaultdict, Counter
import networkx as nx
from placeandroute.SteinerTree import make_steiner_tree
from chimerautils.display import interactive_embeddingview
import os

def fast_steiner_tree(graph, voi):
    ephnode = "ephemeral"
    graph.add_edge(ephnode, voi[0], weight=0)
    for n in voi[1:]:
        path = nx.dijkstra_path(graph, ephnode, n)
        for nn in path[2:]:
            if nn not in graph.neighbors(ephnode):
                graph.add_edge(ephnode, nn, weight=0)
    treenodes = graph.neighbors(ephnode)
    graph.remove_node(ephnode)
    ret =  graph.subgraph(treenodes)
    return nx.minimum_spanning_tree(ret)


class MinMaxRouter(object):
    def __init__(self, graph, terminals):
        self.result = dict()
        self.nodes = terminals.keys()
        self.terminals = terminals
        self.wgraph = graph.copy()
        for n1, n2 in self.wgraph.edges_iter():
            self.wgraph.edge[n1][n2]["weight"] = float(1)

    def increase_weights(self, edges):
        for n1, n2 in edges:
            self.wgraph.edge[n1][n2]["weight"] *= 2000

    def run(self, effort):
        candidatetrees = defaultdict(Counter)
        for _ in xrange(effort):
            for node, terminals in self.terminals.iteritems():
                #newtree = make_steiner_tree(self.wgraph, list(terminals))
                newtree = fast_steiner_tree(self.wgraph, list(terminals))
                candidatetrees[node][newtree] += 1
                self._update_weights(newtree)
        ## compact candidatetrees
        for node in self.terminals.keys():
            ## xxx allow overlapping for now
            self.result[node] = candidatetrees[node].most_common(1)[0][0]

    def _update_weights(self, tree):
        for n1, n2 in tree.edges_iter():
            self.wgraph.edge[n1][n2]["weight"] *= 2000

    def get_tree(self, node):
        return self.result[node]

    def get_cost(self):
        return {n: g.number_of_nodes() for n, g in self.result.iteritems()}

    def is_present(self, p):
        return p in self.result


class EdgeHeuristic(object):
    def __init__(self, problemGraph, archGraph):
        self.problemEdges = problemGraph.edges()  # list of pairs
        self.archEdges = archGraph.edges()  # list of pairs
        self.problemGraph = problemGraph  # nx.Graph
        self.archGraph = archGraph  # nx.Graph
        self.rounds = 1000
        self.assignment = None  # dict from problemEdges to archEdges
        self.tabu = list()

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

    def initial_assignment_bfs(self, coords):
        self.assignment = dict()
        self.find_routing(effort=1)
        self._prefer_center(coords)
        centrality = Counter(nx.closeness_centrality(self.problemGraph))
        startnode = centrality.most_common(1)[0][0]
        ripped = None
        for edge in nx.bfs_edges(self.problemGraph, startnode):
            if ripped is not None:
                ripped = self._rip_edge(effort=5)
            self._insert_edge(edge, effort=5)
            if ripped is not None:
                self._insert_edge(ripped, effort=5)
            else:
                ripped = 1


    def choose_ripped_edge(self):
        node_score = self.routing.get_cost() # use this but weighted node usage not just count
        ret = None
        best_score = 0
        for k, v in self.assignment.iteritems():
            score = self.archGraph[v[0]][v[1]]['weight']
            if ret is None or  score > best_score and k not in self.tabu:
                ret = k
                best_score = score
        if len(self.tabu) == min(5, len(self.assignment)):
            self.tabu.pop(0)
        self.tabu.append(ret)
        return ret

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

        os.environ["PATH"] += ":/home/svarotti/Drive/dwaveproj/projects/embeddingview"
        interactive_embeddingview(self.problemGraph, 16, False)

    def _update_weights(self):
        router = self.routing

        for n1 in self.archGraph.nodes_iter():
            self.archGraph.node[n1]["mappedto"] = set()

        for node in self.problemGraph.nodes_iter():
            if router.is_present(node):
                nn = router.get_tree(node).nodes()
                for nnn in nn:
                    self.archGraph.node[nnn]["mappedto"].add(node)
            else:
                nn = list()
            self.problemGraph.node[node]["mapto"] = nn

        exp = 2000
        getsize = lambda n: len(self.archGraph.node[n]["mappedto"])
        for n1, n2 in self.archGraph.edges_iter():
            self.archGraph[n1][n2]["weight"] = max([exp**getsize(n1), exp**getsize(n2)])


    def choose_best_edge(self, probEdge):
        # xxx remember archedge can be flipped! (b, a)
        self._bestEdge = None
        self._bestDist = None
        self._probEdge = probEdge


        for n1, n2 in self.archEdges:
            self._update_optimum(n1, n2)
            self._update_optimum(n2, n1)

        return self._bestEdge

    def _node_distance(self, p, tonode):
        ephemeral = ("ephemeral node")

        if not self.routing.is_present(p):
            return  len(self.archGraph.node[tonode]["mappedto"])

        nodes = self.routing.get_tree(p).nodes()
        for node in nodes:
            self.archGraph.add_edge(ephemeral, node, weight=0)
        ret =  nx.shortest_path_length(self.archGraph, tonode, ephemeral)
        self.archGraph.remove_node(ephemeral)
        return ret

    def _update_optimum(self, n1, n2):
        p1, p2 = self._probEdge
        dist = self._node_distance(p1, n1) + self._node_distance(p2, n2) \
               + self.archGraph[n1][n2]['distance_from_center'] \
               + 2000 ** len(self.archGraph.node[n1]["mappedto"].difference({p1}))\
               + 2000 ** len(self.archGraph.node[n2]["mappedto"].difference({p2}))
        if self._bestEdge is None or dist < self._bestDist:
            self._bestEdge = (n1, n2)
            self._bestDist = dist

    def run(self):
        if self.assignment is None:
            self.initial_assignment_random()

        for round in xrange(self.rounds):
            choice = self._rip_edge(effort=10)
            self._insert_edge(choice, effort=10)

            print sum(self.routing.get_cost().itervalues()) # bad count
            count = self._edge_usage()
            if count.most_common(1)[0][1] == 1: # break on node usage
                break

    def _edge_usage(self):
        count = Counter()
        for tree in self.routing.result.values():
            count.update(map(frozenset,tree.edges()))
        for aed in self.assignment.values():
            count[frozenset(aed)] += 1
        return count

    def _insert_edge(self, choice, effort):
        dest = self.choose_best_edge(choice)
        self.assignment[choice] = dest
        self.find_routing(effort)

    def _rip_edge(self, effort):
        choice = self.choose_ripped_edge()
        del self.assignment[choice]
        self.find_routing(effort)
        return choice

