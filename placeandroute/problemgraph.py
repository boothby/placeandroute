from itertools import groupby
from math import copysign


def sign(x):
    return copysign(1, x)


def parse_cnf(f):
    ret = []
    for line in f:
        if line.startswith("p") or line.startswith('c'):
            continue
        lits0 = line.strip().split()
        ret.append(tuple(int(x) for x in lits0[:-1]))
    return ret


def num_vars(cnf):
    return max(max(abs(l) for l in cl) for cl in cnf)


def clause_to_hubo(cl):
    ret = []
    ret.append((1, set()))
    ret.append((sign(cl[0]), {abs(cl[0])}))
    for l in cl[1:]:
        newret = []
        for coeff, terms in ret:
            newret.append((coeff * sign(l), terms ^ {abs(l)}))
        ret.extend(newret)

    return [(sum(a[0] for a in g), k) for k, g in groupby(ret, lambda x: x[1])]


from collections import Counter, namedtuple
from itertools import combinations

iff = namedtuple('iff', ('f', 'x1', 'x2'))


def decrease_width(clauses):
    global lastx1, lastx2
    nvars = num_vars(clauses)

    scoreboard = Counter()

    for cl in clauses:
        if type(cl) == iff: continue
        if len(cl) < 3: continue
        for cnt in combinations(cl, 2):
            scoreboard.update([cnt])

    winner = scoreboard.most_common(1)[0][0]
    newvar = nvars + 1
    clauses.append(iff(newvar, winner[0], winner[1]))
    for cl in clauses:
        if type(cl) == iff:
            continue
        elif all(x in cl for x in winner):
            for x in winner:
                cl.remove(x)
            cl.append(newvar)
        else:
            continue

    return clauses


def cnf_to_aig(clauses):
    clauses = list(clauses)
    while max(len(x) for x in clauses if type(x) != iff) > 2:
        clauses = decrease_width(clauses)
    return clauses


def cnf_to_qubo(clauses):
    clauses = list(list(x) for x in clauses)
    while max(len(x) for x in clauses if type(x) != iff) > 2:
        clauses = decrease_width(clauses)
    ret = []
    for cl in clauses:
        if type(cl) == iff:
            c, a, b = cl
            ret.extend([
                (0.5 * 4, set()),
                (0.5 * 1 * sign(a), {abs(a)}),
                (0.5 * 2 * sign(b), {abs(b)}),
                (0.5 * -3 * sign(c), {abs(c)}),
                (0.5 * -2 * sign(a) * sign(c), {abs(a), abs(c)}),
                (0.5 * -3 * sign(b) * sign(c), {abs(b), abs(c)}),
                (0.5 * 1 * sign(a) * sign(b), {abs(a), abs(b)}),
            ])  # XXX
        elif len(cl) < 2:
            ret.extend([
                (1, set()),
                (-1 * sign(cl[0]), {abs(cl[0])})
            ]
            )
        else:
            a, b = cl
            ret.extend([
                (0.5 * 1, set()),
                (0.5 * -1 * sign(a), {abs(a)}),
                (0.5 * -1 * sign(b), {abs(b)}),
                (0.5 * 1 * sign(a) * sign(b), {abs(a), abs(b)})
            ])
    retdict = dict()
    for c, term in ret:
        term = frozenset(term)
        if term not in retdict:
            retdict[term] = c
        else:
            retdict[term] += c
    return [(b, a) for a, b in retdict.items()]


def cnf_to_hubo(clauses):
    ret = []
    for cl in clauses:
        ret.extend(clause_to_hubo(cl))
    return [(sum(a[0] for a in g), k) for k, g in groupby(ret, lambda x: x[1])]


import networkx as nx


def cnf_to_graph(clauses):
    qubo = cnf_to_qubo(clauses)
    ret = nx.Graph()
    for val, term in qubo:
        if len(term) == 2:
            node1, node2 = term
            ret.add_edge(node1, node2)
    return ret
