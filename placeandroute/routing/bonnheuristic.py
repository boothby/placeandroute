import random as rand

import networkx as nx
from collections import defaultdict, Counter


def fast_steiner_tree(graph, voi):
    ephnode = "ephemeral"
    graph.add_edge(ephnode, voi[0], weight=0)
    for n in voi[1:]:
        path = nx.astar_path(graph, ephnode, n)
        for nn in path[2:]:
            if nn not in graph.neighbors(ephnode):
                graph.add_edge(ephnode, nn, weight=0)
    treenodes = graph.neighbors(ephnode)
    graph.remove_node(ephnode)
    return frozenset(treenodes)
    #ret =  graph.subgraph(treenodes)
    #return nx.minimum_spanning_tree(ret)


def choice_weighted(l):
    r = rand.random()
    for t, prob in l:
        if r < prob:
            return t
        else:
            r -= prob
    return l[-1]


class MinMaxRouter(object):
    def __init__(self, graph, terminals):
        self.result = dict()
        self.nodes = terminals.keys()
        self.terminals = terminals
        self.wgraph = nx.Graph(graph) #shallow copy
        for n1, n2 in self.wgraph.edges_iter():
            self.wgraph.edge[n1][n2]["weight"] = float(1)

    def increase_weights(self, edges):
        for n1, n2 in edges:
            self.wgraph.edge[n1][n2]["weight"] *= 2000

    def run(self, effort):
        candidatetrees = defaultdict(Counter)
        if not self.terminals:
            return

        for _ in xrange(effort):
            for node, terminals in self.terminals.iteritems():
                #newtree = make_steiner_tree(self.wgraph, list(terminals))
                newtree = fast_steiner_tree(self.wgraph, list(terminals))
                candidatetrees[node][newtree] += 1
                self._update_weights(newtree)
        ## compact candidatetrees
        for node in self.terminals.keys():
            ## xxx allow overlapping for now
            self.result[node] = [(t, float(q)/effort) for t,q in candidatetrees[node].most_common()]
        self.derandomize(effort)


    def derandomize(self,effort):
        ret = dict()
        nodes = self.terminals.keys()
        for node in nodes:
            c = self.result[node][0][0]
            ret[node] = c

        def invertres(r):
            ret = defaultdict(list)
            for k,vv in r.iteritems():
                for v in vv:
                    ret[v].append(k)
            return ret

        def calcscore(r):
            return sum(float(5)**len(x) for x in invertres(r).itervalues())

        score = calcscore(ret)
        for _ in range(effort):
          for node in nodes:
            c = choice_weighted(self.result[node])
            oldc = ret[node]
            if c == oldc: continue
            ret[node] = c
            newscore = calcscore(ret)
            if newscore < score:
                score = newscore
            else:
                ret[node] = oldc

        self.result = ret



    def _update_weights(self, nodes):
        for n1, n2 in self.wgraph.subgraph(nodes).edges_iter():
            self.wgraph.edge[n1][n2]["weight"] *= 2000

    def get_tree(self, node):
        return self.wgraph.subgraph(self.result[node])

    def get_nodes(self, node):
        return self.result[node]

    def get_cost(self):
        return {n: len(g) for n, g in self.result.iteritems()}

    def is_present(self, p):
        return p in self.result