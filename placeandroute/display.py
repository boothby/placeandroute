import json
import os

import networkx as nx


def interactive_embeddingview(G, size):
    def reorder (x):
        return (x//8)*8 + ((x+4) % 8)
    embedding = [ list(reorder(x) for x in data["mapto"]) for _, data in G.nodes_iter(data=True)]
    #embedding = [ [x] for x in range(0,100,8)]
    tfn = os.tmpnam() + ".json"
    paras = "--solver chimera {0} {0} 4 --embedding {1} ".format(size, tfn)
    #print json.dumps(embedding)
    try:
        with open(tfn, "w") as f:
            f.write(json.dumps(embedding))
        os.system("/home/svarotti/Drive/dwaveproj/projects/embeddingview/EmbeddingView " + paras + " &")
        raw_input("pause")
    finally:
        pass #os.unlink(tfn)


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
    #nx.draw_networkx_edges(g,layout, width=[log(x["weight"],30) for _,_,x in g.edges_iter(data=True)])
    plt.savefig(tfn)
    os.system("feh " + tfn)


def display_graphml(g,l):
    tfn = os.tmpnam() + ".graphml"
    try:
        with open(tfn, "w") as f:
            nx.write_graphml(g, f)

        os.system("gephi " + tfn)
    finally:
        os.unlink(tfn)


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