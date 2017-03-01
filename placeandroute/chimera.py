import networkx as nx


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