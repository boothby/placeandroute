from itertools import combinations, chain
import networkx as nx



def updateVertexWeight(node, aGraph, exponent=30):
    aGraph.node[node]["score"] = 0
    for nnode in aGraph.neighbors_iter(node):
        aGraph.edge[nnode][node]["weight"] *= exponent

def steinerTree(archGraph, voi):
    steinerGraph = nx.Graph()
    for v1, v2 in  combinations(voi, 2):
        paths = nx.all_shortest_paths(archGraph, v1, v2, "weight")
        paths2 = (nx.all_shortest_paths(archGraph, v2,v1, "weight"))
        for path in chain(paths, paths2):
            for a, b in zip(path[:-1], path[1:]):
                if not steinerGraph.has_edge(a,b):
                    steinerGraph.add_edge(a,b, weight= archGraph.edge[a][b]["weight"])

    return nx.minimum_spanning_tree(steinerGraph,"weight")

from SteinerTree import make_steiner_tree as steinerTree
from random import shuffle

def embed(problemGraph, archGraph):
    #partEmbedding = Chainmap()
    for e1,e2 in archGraph.edges_iter():
        archGraph.edge[e1][e2]["weight"] = 1.0

    coords = ((data["x"], data["y"]) for _,data in archGraph.nodes_iter(data=True))
    sumcoords = reduce(lambda (xs,ys), (x,y): (xs+x, ys+y), coords)
    avgx, avgy = map(lambda x: x/archGraph.number_of_nodes(), sumcoords)

    oldArchGraph = archGraph
    archGraph = archGraph.to_directed()


    for vertex in problemGraph.nodes_iter():
        problemGraph.node[vertex]["mapto"] = []

    for vertex in archGraph.nodes_iter():
        archGraph.node[vertex]["mapped"] = []

    mappingorder = problemGraph.nodes()
    shuffle(mappingorder)

    for nextMappedNode in mappingorder:
        alreadyMappedNeighs = [x for x in problemGraph.neighbors_iter(nextMappedNode)
                              if len(problemGraph.node[x]["mapto"]) > 0]
        for vertex,data in archGraph.nodes_iter(data=True):
            archGraph.node[vertex]["score"] = (abs( data["x"] - avgx) + abs(data["y"] - avgy)
                                              # (1.0*(7-archGraph.degree(vertex)) )
                                            + 3000.0**len(data["mapped"]))
            #archGraph.node[vertex]["pred"] = dict()

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
                    archGraph.node[node]["score"] = 1e10
                    #archGraph.node[node]["pred"][mappedNeigh] = None
                    continue
                archGraph.node[node]["score"] += score
                #archGraph.node[node]["pred"][mappedNeigh] = pred
            #remove node
            archGraph.remove_node(ephnode)

        #find minscore
        minnode = None
        for chooseme in archGraph.nodes_iter():
            #if chooseme == nextMappedNode:continue
            if minnode is None or archGraph.node[chooseme]["score"] < archGraph.node[minnode]["score"]:
                minnode = chooseme

        voi = [minnode]
        for mappedNeigh in alreadyMappedNeighs:
            #add ephnode
            ephnode = ("eph", mappedNeigh)
            voi.append(ephnode)
            archGraph.add_node(ephnode)
            termnodes = set()
            mappedChain = problemGraph.node[mappedNeigh]["mapto"]
            for vertex in mappedChain:
                termnodes.update(archGraph.neighbors(vertex))
            termnodes.difference_update(voi)
            termnodes.difference_update(mappedChain)
            for vertex in termnodes:
                highnum = 1e10
                archGraph.add_edge(ephnode, vertex, weight=highnum)
                archGraph.add_edge(vertex,ephnode, weight=highnum)

        if len(voi) > 1:
            newchain = steinerTree(archGraph, voi) # type: nx.Graph
            for ephnode in voi[1:]:
                archGraph.remove_node(ephnode)
                #newchain.remove_nodes_from(newchain.neighbors(ephnode))
                newchain.remove_node(ephnode)
            assert minnode in newchain
            assert nx.is_connected(oldArchGraph.subgraph(newchain.nodes_iter()))
        else:
            newnode = minnode
            newchain = nx.Graph()
            newchain.add_node(newnode)

        chains = set(newchain.nodes_iter())
        #debug_newneigh = set()
        for newnode in chains:
            #assert any(x not in debug_newneigh for x in archGraph.neighbors(newnode))
            #debug_newneigh.update(archGraph.neighbors(newnode))
            updateVertexWeight(newnode, archGraph)
            archGraph.node[newnode]["mapped"].append(nextMappedNode)
        problemGraph.node[nextMappedNode]["mapto"] = chains
        print nextMappedNode, chains

    for node in oldArchGraph.nodes_iter():
        oldArchGraph.node[node]["mapped"] = archGraph.node[node]["mapped"]
