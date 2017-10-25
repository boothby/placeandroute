from random import random

import networkx as nx

from placeandroute.SteinerTree import make_steiner_tree as steinerTree


def updateVertexWeight(node, aGraph, exponent):
    # type: (int, nx.Graph, int) -> None
    for nnode in aGraph.neighbors_iter(node):
        aGraph.edge[nnode][node]["weight"] *= exponent


def embed(problemGraph, archGraph):

    #expo_param = nx.diameter(archGraph) * 2
    expo_param = archGraph.number_of_nodes()

    initialize_embedding(archGraph, problemGraph)

    avgx, avgy = get_center(archGraph)

    oldArchGraph = archGraph
    archGraph = archGraph.to_directed()

    mappingorder = initialize_mapping_order(problemGraph)

    for nextMappedNode in mappingorder:
        for vertex,data in archGraph.nodes_iter(data=True):
            for nvertex in (archGraph.neighbors_iter(vertex)):
                assert archGraph.edge[nvertex][vertex]["weight"] == (expo_param ** len(data["mapped"]))

        alreadyMappedNeighs = [x for x in problemGraph.neighbors_iter(nextMappedNode)
                              if len(problemGraph.node[x]["mapto"]) > 0]
        for vertex,data in archGraph.nodes_iter(data=True):
            archGraph.node[vertex]["score"] = (abs( data["x"] - avgx) + abs(data["y"] - avgy)
                                              # (1.0*(7-archGraph.degree(vertex)) )
                                            #+ expo_param**len(data["mapped"])
                                            ) * 1 + random()**2
            #archGraph.node[vertex]["pred"] = dict()

        for mappedNeigh in alreadyMappedNeighs:
            #add ephnode
            ephnode = "xxx"
            archGraph.add_node(ephnode);
            neighset = set(problemGraph.node[mappedNeigh]["mapto"])
            for vertex in problemGraph.node[mappedNeigh]["mapto"]:
                neighset.update(archGraph.neighbors_iter(vertex))

            for nvertex in neighset:
                    archGraph.add_edge(ephnode, nvertex, weight=expo_param**len(archGraph.node[nvertex]["mapped"]))

            #calc & update score
            distances = nx.single_source_dijkstra_path_length(archGraph, ephnode, weight="weight")
            for node in archGraph.nodes_iter():
                if node == ephnode: continue
                #pred = preds[node]
                score = distances[node]
                #if ephnode in pred:
                #    archGraph.node[node]["score"] = 1e100
                #    #archGraph.node[node]["pred"][mappedNeigh] = None
                #    continue
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

        ephnode0 = ("eph0", minnode)
        archGraph.add_edge(ephnode0, minnode, weight=0)
        archGraph.add_edge(minnode, ephnode0, weight=0)
        voi = [ephnode0]
        for mappedNeigh in alreadyMappedNeighs:
            #add ephnode
            ephnode = ("eph", mappedNeigh)
            voi.append(ephnode)
            archGraph.add_node(ephnode)
            termnodes = set()
            mappedChain = problemGraph.node[mappedNeigh]["mapto"]
            for vertex in mappedChain:
                termnodes.update(archGraph.neighbors(vertex))
            termnodes.difference_update(voi) # xxx minnode
            termnodes.difference_update(mappedChain)
            for vertex in termnodes:
                highnum = 1e50
                archGraph.add_edge(ephnode, vertex, weight=highnum)
                archGraph.add_edge(vertex,ephnode, weight=highnum)


        if len(voi) > 1:
            newchain = steinerTree(archGraph, voi) # type: nx.Graph
            for ephnode in voi:
                archGraph.remove_node(ephnode)
                #newchain.remove_nodes_from(newchain.neighbors(ephnode))
                newchain.remove_node(ephnode)
            assert minnode in newchain
            assert nx.is_connected(oldArchGraph.subgraph(newchain.nodes_iter()))
        else:
            archGraph.remove_node(ephnode0)
            newnode = minnode
            newchain = nx.Graph()
            newchain.add_node(newnode)

        chains = set(newchain.nodes_iter())
        #debug_newneigh = set()
        for newnode in chains:
            #assert any(x not in debug_newneigh for x in archGraph.neighbors(newnode))
            #debug_newneigh.update(archGraph.neighbors(newnode))
            updateVertexWeight(newnode, archGraph, expo_param)
            archGraph.node[newnode]["mapped"].append(nextMappedNode)
        problemGraph.node[nextMappedNode]["mapto"] = chains
        print nextMappedNode, chains
        #interactive_embeddingview(problemGraph, 16)

    for _ in range(100):
        #interactive_embeddingview(problemGraph, 16, stop=False)
        reencode(archGraph, avgx, avgy, mappingorder, oldArchGraph, problemGraph, expo_param)





    for node in oldArchGraph.nodes_iter():
        oldArchGraph.node[node]["mapped"] = archGraph.node[node]["mapped"]


def initialize_mapping_order(problemGraph):
    mappingorder = problemGraph.nodes()
    ncentrality = nx.closeness_centrality(problemGraph)
    mappingorder.sort(key=ncentrality.get)
    mappingorder.reverse()
    mappingorder = nx.topological_sort(nx.bfs_tree(problemGraph, mappingorder[0]), mappingorder)
    # shuffle(mappingorder)
    return mappingorder


def initialize_embedding(archGraph, problemGraph):
    for e1, e2 in archGraph.edges_iter():
        archGraph.edge[e1][e2]["weight"] = 1.0
    for vertex in problemGraph.nodes_iter():
        problemGraph.node[vertex]["mapto"] = []
    for vertex in archGraph.nodes_iter():
        archGraph.node[vertex]["mapped"] = []


def get_center(archGraph):
    coords = ((data["x"], data["y"]) for _, data in archGraph.nodes_iter(data=True))
    sumcoords = reduce(lambda (xs, ys), (x, y): (xs + x, ys + y), coords)
    avgx, avgy = map(lambda x: x / archGraph.number_of_nodes(), sumcoords)
    return avgx, avgy


def reencode(archGraph, avgx, avgy, mappingorder, oldArchGraph, problemGraph, expo_param):
    #shuffle(mappingorder)
    mappingorder.sort(key=lambda n: -sum(expo_param ** len(archGraph.node[anode]["mapped"]) for anode in problemGraph.node[n]["mapto"]))
    for nextMappedNode in mappingorder:
        for mappedto in problemGraph.node[nextMappedNode]["mapto"]:
            archGraph.node[mappedto]["mapped"].remove(nextMappedNode)
            for neigmapped in archGraph.neighbors_iter(mappedto):
                archGraph.edge[neigmapped][mappedto]["weight"] /= expo_param
        problemGraph.node[nextMappedNode]["mapto"] = set()

        assert all(all(
            (archGraph.edge[nvertex][vertex]["weight"] == (expo_param ** len(data["mapped"])))
            for nvertex in (archGraph.neighbors_iter(vertex)))
                   for vertex, data in archGraph.nodes_iter(data=True))
        alreadyMappedNeighs = [x for x in problemGraph.neighbors_iter(nextMappedNode)
                               if len(problemGraph.node[x]["mapto"]) > 0]
        for vertex, data in archGraph.nodes_iter(data=True):
            archGraph.node[vertex]["score"] = 0*(expo_param** len(data["mapped"]))
            # archGraph.node[vertex]["pred"] = dict()

        for mappedNeigh in alreadyMappedNeighs:
            # add ephnode
            ephnode = "xxx"
            archGraph.add_node(ephnode)

            neighset = set(problemGraph.node[mappedNeigh]["mapto"])
            for vertex in problemGraph.node[mappedNeigh]["mapto"]:
                neighset.update(archGraph.neighbors_iter(vertex))

            for nvertex in neighset:
                archGraph.add_edge(ephnode, nvertex, weight=expo_param ** len(archGraph.node[nvertex]["mapped"]))

            # calc & update score
            distances = nx.single_source_dijkstra_path_length(archGraph, ephnode, weight="weight")
            for node in archGraph.nodes_iter():
                if node == ephnode: continue
                score = distances[node]

                archGraph.node[node]["score"] += score
            # remove node
            archGraph.remove_node(ephnode)

        # find minscore
        minnode = None
        for chooseme in archGraph.nodes_iter():
            # if chooseme == nextMappedNode:continue
            if minnode is None or archGraph.node[chooseme]["score"] < archGraph.node[minnode]["score"]:
                minnode = chooseme

        ephnode0 = ("eph0", minnode)
        archGraph.add_edge(ephnode0, minnode, weight=0)
        archGraph.add_edge(minnode, ephnode0, weight=0)
        voi = [ephnode0]
        for mappedNeigh in alreadyMappedNeighs:
            # add ephnode
            ephnode = ("eph", mappedNeigh)
            voi.append(ephnode)
            archGraph.add_node(ephnode)
            termnodes = set()
            mappedChain = problemGraph.node[mappedNeigh]["mapto"]
            for vertex in mappedChain:
                termnodes.update(archGraph.neighbors(vertex))
            termnodes.difference_update(voi)  # xxx minnode
            termnodes.difference_update(mappedChain)
            for vertex in termnodes:
                highnum = 1e50
                archGraph.add_edge(ephnode, vertex, weight=highnum)
                archGraph.add_edge(vertex, ephnode, weight=highnum)

        if len(voi) > 1:
            newchain = steinerTree(archGraph, voi)  # type: nx.Graph
            for ephnode in voi:
                archGraph.remove_node(ephnode)
                # newchain.remove_nodes_from(newchain.neighbors(ephnode))
                newchain.remove_node(ephnode)
            assert minnode in newchain
            assert nx.is_connected(oldArchGraph.subgraph(newchain.nodes_iter()))
        else:
            archGraph.remove_node(ephnode0)
            newnode = minnode
            newchain = nx.Graph()
            newchain.add_node(newnode)

        chains = set(newchain.nodes_iter())
        # debug_newneigh = set()
        for newnode in chains:
            # assert any(x not in debug_newneigh for x in archGraph.neighbors(newnode))
            # debug_newneigh.update(archGraph.neighbors(newnode))
            updateVertexWeight(newnode, archGraph, expo_param)
            archGraph.node[newnode]["mapped"].append(nextMappedNode)
        problemGraph.node[nextMappedNode]["mapto"] = chains
        print nextMappedNode, chains
        # interactive_embeddingview(problemGraph, 16)