import networkx as nx
from typing import List, Dict


def chimeratiles(w,h):
    # type: (int, int) -> (nx.Graph, List[int])
    """Return a compressed version of a chimera graph, where each row in a tile is merged in a single node"""
    ret = nx.Graph()
    i = 0
    grid = []
    choices = []
    for c in range(w):
        row = []
        for r in range(h):
            n1, n2 = i, i +1
            i += 2
            ret.add_nodes_from([n1,n2], capacity=4, usage=0)
            ret.add_edge(n1,n2, capacity=16)
            if row:
                ret.add_edge(n1, row[-1][0], capacity=4)
            if grid:
                ret.add_edge(n2,grid[-1][len(row)][1],capacity=4)
            row.append((n1,n2))
            choices.extend([(n1,n2), (n2,n1)])
        grid.append(row)
    return ret, choices


def expand_solution(g, chains, cg):
    # type: (nx.Graph, Dict[int, List[int]], nx.Graph) -> Dict[int, List[int]]
    x = dict()
    alreadymapped = set()
    assert all(cg.has_edge(n1 * 4, n2 * 4) for n1, n2 in g.edges())
    for k, v in sorted(chains.items(), key=lambda (k,v): -len(v)):
        # single node case this is easy
        if len(v) == 1:
            nodegroup = next(iter(v))
            choice = range(nodegroup * 4, (nodegroup+1)*4)
            choice = filter(lambda  x: x not in alreadymapped, choice)[:1]
            alreadymapped.update(choice)
            x[k]= choice
            continue

        #long chain. rationale: build the subgraph of possible chains, then trim it
        #first filter already used nodes, then pick a connected component that has at least one choice
        # for each group, trim the choices slowly
        vs = []
        for nodegroup in v:
            choices = range(nodegroup * 4, (nodegroup + 1) * 4)
            choices = filter(lambda x: x not in alreadymapped, choices)
            assert choices
            vs.append(choices)
        allchoices = reduce(list.__add__, vs)
        sg = cg.subgraph(allchoices)  # type: nx.Graph
        chosen_sg = None
        for component in nx.connected_components(sg):
            #gcomponent = sg.subgraph(component)
            if all(any(n in component for n in choice) for choice in vs) and chosen_sg is None:
                chosen_sg = component
            else:
                for n in component:
                    allchoices.remove(n)
                    for v in vs:
                        if n in v: v.remove(n)
        assert chosen_sg is not None, (sg.nodes(), sg.edges())
        sg = sg.subgraph(chosen_sg)
        sg = nx.minimum_spanning_tree(sg)
        assert nx.is_connected(sg), (list(nx.connected_components(sg)), allchoices, vs)
        noprogress = 0
        while any(len(v) > 1 for v in vs):
            noprogress += 1
            for v in vs:
                if len(v) == 1: continue
                if not all(n in sg.nodes() for n in v):
                    raise Exception
                nn = filter(lambda n: sg.degree(n) == 1, v)
                if len(nn) == len(v): nn.pop(0)
                if nn:
                    for n in nn:
                        sg.remove_node(n)
                        allchoices.remove(n)
                        v.remove(n)
                    noprogress = 0
            if noprogress > 1000:
                for v in vs:
                    if len(v) > 1:
                        n = v.pop(0)
                        sg.remove_node(n)
                        allchoices.remove(n)
                        noprogress=0
                        break

        vs = [v[0] for v in vs]
        alreadymapped.update(vs)
        x[k] = vs
    return x