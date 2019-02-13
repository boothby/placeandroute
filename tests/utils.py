from re import split

from placeandroute.tilebased.heuristic import Constraint


def cnf_to_constraints(clauses, num_vars):
    ancilla = num_vars + 1
    for clause in clauses:
        yield Constraint([map(abs, clause[:2]),[abs(clause[2]), ancilla]])
        ancilla += 1


def parse_2in4(f):
    ret = []
    for line in f:
        if not line.startswith("TWOINFOUR"): continue
        res = list(map(int, split(r'[^0-9]+', line)[1:-1]))
        assert len(res) == 4, res
        a, b, c, d = res
        ret.append([[a,b],[c,d]])
    return ret