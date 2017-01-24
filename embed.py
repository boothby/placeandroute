
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
