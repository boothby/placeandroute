#!/usr/bin/env python2

# formula generated with seed 1, probabilty values 0.414000 0.028000 0.503000 and clause
#ration 4.408571


p_i = [0, 0.414000, 0.028000, 0.503000 ]
r = 4.408571
import sys

f = []
nvars = int(sys.argv[1])

import random
from math import copysign
sol = [copysign(i, random.choice([-1,1])) for i in range(1, nvars+1)] # xxx

def gen_rand_clause(nvars):
    base = range(-nvars, 0) + range(1, nvars+1)
    return random.sample(base, 3)

def numofsatlit(cl, sol):
    return sum( sol.count(x) for x in cl)

while len(f) < nvars * r:
    cl = gen_rand_clause(nvars)
    i = numofsatlit(cl, sol)
    if random.random() < p_i[i]:
        f.append(cl)
print 'c', nvars, r, p_i
print 'c', sol
print 'p cnf', nvars, len(f)
for cl in f:
    print ' '.join(map(str, cl)), 0