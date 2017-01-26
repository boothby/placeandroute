import networkx as nx
from collections import defaultdict
#DICT[var, [qubits]] chainmap
#DICT[qubit, var] assignment

class Chainmap(defaultdict):
    def __init__(self):
        defaultdict.__init__(self, list)

    def toAssignment(self):
        ret = Assignment()
        for k,vs in self:
            for v in vs:
                assert v not in ret
                ret[v] = k
        return ret

class Assignment(dict):
    def toChainmap(self):
        ret = Chainmap()
        for k,v in self:
            ret[k].append(v)
        return ret


def embed(problemGraph, archGraph):
    #partEmbedding = Chainmap()
    for e1,e2 in archGraph.edges_iter():
        archGraph.edge[e1][e2]["weight"] = 1.0

    oldArchGraph = archGraph
    archGraph = archGraph.to_directed()


    for vertex in problemGraph.nodes_iter():
        problemGraph.node[vertex]["mapto"] = []

    for vertex in archGraph.nodes_iter():
        archGraph.node[vertex]["mapped"] = []

    for nextMappedNode in problemGraph.nodes_iter():
        alreadyMappedNeighs = [x for x in problemGraph.neighbors_iter(nextMappedNode)
                              if len(problemGraph.node[x]["mapto"]) > 0]
        for vertex in archGraph.nodes_iter():
            archGraph.node[vertex]["score"] = (abs(vertex - (archGraph.number_of_nodes()/2) )
                                              # (1.0*(7-archGraph.degree(vertex)) )
                                            + 100.0*len(archGraph.node[vertex]["mapped"]))
            archGraph.node[vertex]["pred"] = dict()

        for mappedNeigh in alreadyMappedNeighs:
            #add ephnode
            ephnode = "xxx"
            archGraph.add_node(ephnode);
            for vertex in problemGraph.node[mappedNeigh]["mapto"]:
                archGraph.add_edge(ephnode, vertex, weight=0)

            #calc & update score
            preds, distances = nx.dijkstra_predecessor_and_distance(archGraph, ephnode)
            for node in archGraph.nodes_iter():
                if node == ephnode: continue
                pred = preds[node]
                score = distances[node]
                if ephnode in pred:
                    archGraph.node[node]["score"] = 1e100
                    archGraph.node[node]["pred"][mappedNeigh] = None
                    continue
                archGraph.node[node]["score"] += score
                archGraph.node[node]["pred"][mappedNeigh] = pred
            #remove node
            archGraph.remove_node(ephnode)

        #find minscore
        minnode = None
        for chooseme in archGraph.nodes_iter():
            if chooseme == nextMappedNode:continue
            if minnode is None or archGraph.node[chooseme]["score"] < archGraph.node[minnode]["score"]:
                minnode = chooseme

        exponent = 3000
        def updateVertexWeight(node, aGraph, exponent=exponent):
            aGraph.node[node]["score"] = 0
            for nnode in aGraph.neighbors_iter(node):
                aGraph.edge[nnode][node]["weight"] *= exponent

        #update weights and map
        chains = set([minnode])
        archGraph.node[minnode]["mapped"].append(nextMappedNode)
        updateVertexWeight(minnode, archGraph)
        for chainme in alreadyMappedNeighs:
            pred = min(archGraph.node[minnode]["pred"][chainme], key= lambda x:archGraph.node[x]["score"])
            #archGraph.edge[pred][minnode]["weight"] *= exponent
            endnodes = problemGraph.node[chainme]["mapto"]
            while pred not in endnodes:
                archGraph.node[pred]["mapped"].append(nextMappedNode)
                chains.add(pred)
                updateVertexWeight(pred, archGraph)
                allpreds = archGraph.node[pred]["pred"][chainme]
                nextpred = min(archGraph.node[pred]["pred"][chainme], key= lambda x:archGraph.node[x]["score"])
                #archGraph.edge[pred][nextpred]["weight"] *= exponent
                pred = nextpred
        problemGraph.node[nextMappedNode]["mapto"] = chains
        print nextMappedNode, chains

    for node in oldArchGraph.nodes_iter():
        oldArchGraph.node[node]["mapped"] = archGraph.node[node]["mapped"]

from embed import embed

def translate(lst, off):
    ox, oy = off
    return ( (x + ox, y + oy) for x,y in lst)

def create(r,c):
    g = nx.Graph()
    g.add_nodes_from(range(r*c*8))
    l = []
    for i in xrange(r):
        for j in xrange(c):
            cellLayout = ([(2.5, 1), (2.5, 2), (2.5, 3), (2.5, 4),
             (1, 2.5), (2, 2.5), (3, 2.5), (4, 2.5)])
            cellLayout = translate(cellLayout, (j*4.5, i*4.5))
            l.extend(cellLayout)
            x = (i * c + j)*8
            g.add_edges_from([
                (x, x + 4), (x, x + 5), (x, x + 6), (x, x + 7),
                (x + 1, x + 4), (x + 1, x + 5), (x + 1, x + 6), (x + 1, x + 7),
                (x + 2, x + 4), (x + 2, x + 5), (x + 2, x + 6), (x + 2, x + 7),
                (x + 3, x + 4), (x + 3, x + 5), (x + 3, x + 6), (x + 3, x + 7),
            ])

            if j > 0:
                g.add_edges_from([(x, x  - 8), (x+1, x  - 7), (x+2, x  - 6),(x+3, x  - 5)])
            if i > 0:
                g.add_edges_from([(x+4, x  - 8*c+4), (x+5, x  - 8*c+5), (x+6, x  - 8*c + 6),(x+7, x  - 8*c + 7)])
    for i, pos in enumerate(l):
        x, y = tuple(pos)
        g.node[i]["x"] = x
        g.node[i]["y"] = y
        g.node[i]["theta"] = 1
    return g, l


def create2(r,c):
    g = nx.Graph()
    g.add_nodes_from(range(r*c*8))
    l = []
    for i in xrange(r):
        for j in xrange(c):
            cellLayout = ([(0, 1), (0, 2), (0, 3), (0, 4),
             (1.5, 0), (2.5, 0), (3.5, 0), (0.5, 0)])
            cellLayout = translate( cellLayout, (j*5, i*5))
            l.extend(cellLayout)
            x = (i * c + j)*8
            g.add_edges_from([
                (x, x + 4), (x, x + 5), (x, x + 6), (x, x + 7),
                (x + 1, x + 4), (x + 1, x + 5), (x + 1, x + 6), (x + 1, x + 7),
                (x + 2, x + 4), (x + 2, x + 5), (x + 2, x + 6), (x + 2, x + 7),
                (x + 3, x + 4), (x + 3, x + 5), (x + 3, x + 6), (x + 3, x + 7),
                (x, x+1), (x+2, x+3), (x+4, x+5), (x+6, x+7)
            ])

            if j > 0:
                g.add_edges_from([(x, x  - 8), (x+1, x  - 7), (x+2, x  - 6),(x+3, x  - 5)])
            if i > 0:
                g.add_edges_from([(x+4, x  - 8*c+4), (x+5, x  - 8*c+5), (x+6, x  - 8*c + 6),(x+7, x  - 8*c + 7)])

            if j > 0 and i < r-1:
                southwest = c*8 - 8
                g.add_edges_from([(x+4, x + southwest), (x+4, x + southwest + 1),
                             (x+5, x + southwest), (x+5, x + southwest + 1) ])
    for i, pos in enumerate(l):
        x, y = tuple(pos)
        g.node[i]["x"] = float(x)
        g.node[i]["y"] = float(y)
        g.node[i]["theta"] = 1
    return g, l

def display(g,l):
    #v = networkx_viewer.Viewer(g)
    #v.canvas.create_layout = lambda g, scale ,min_distance: dict(zip(g, np.array(l)))
    #v.canvas.replot()
    #v.mainloop()
    #return

    import matplotlib
    matplotlib.use("Cairo")
    import matplotlib.pyplot as plt
    tfn = os.tmpnam() + ".png"

    layout = dict(zip(range(len(l)), l))
    nx.draw_networkx_nodes(g, layout, node_size=[len(x["mapped"]) for _, x in g.nodes_iter(data=True)])
    from math import log
    #nx.draw_networkx_edges(g,layout, width=[log(x["weight"],30) for _,_,x in g.edges_iter(data=True)])
    plt.savefig(tfn)
    os.system("feh " + tfn)

import os

def display_graphml(g,l):
    tfn = os.tmpnam() + ".graphml"
    try:
        with open(tfn, "w") as f:
            nx.write_graphml(g, f)

        os.system("gephi " + tfn)
    finally:
        os.unlink(tfn)
import json
def display_embeddingview(G, size):
    def reorder (x):
        return (x//8)*8 + ((x+4) % 8)
    embedding = [ list(reorder(x) for x in data["mapto"]) for _, data in G.nodes_iter(data=True)]
    #embedding = [ [x] for x in range(0,100,8)]
    tfn = os.tmpnam() + ".json"
    paras = "--solver chimera {0} {0} 4 --embedding {1} ".format(size, tfn)
    print json.dumps(embedding)
    try:
        with open(tfn, "w") as f:
            f.write(json.dumps(embedding))
        os.system("/home/svarotti/Drive/dwaveproj/projects/embeddingview/EmbeddingView " + paras)
    finally:
        os.unlink(tfn)

if __name__ == '__main__':
    G = nx.Graph()
    G.add_nodes_from([1,2])
    G.add_edges_from([(1,2)])
    A = nx.Graph()
    A.add_nodes_from([1,2])
    A.add_edge(1,2)
    G, _ = create2(2,2)
    A, l = create(8,8)
    embed(G,A)
    for vertex in A.nodes_iter():
        if "pred" in A.node[vertex]:
            del A.node[vertex]["pred"]
        A.node[vertex]["mapped"] = str(A.node[vertex]["mapped"])
    print A.node
    display_embeddingview(G, 8)
    #display(A, l)


