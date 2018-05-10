from os.path import dirname

from six import iteritems, print_

from placeandroute.tilebased.heuristic import Constraint
from placeandroute.tilebased.parallel import ParallelPlacementHeuristic
from placeandroute.tilebased.chimera_tiles import chimeratiles, expand_solution
from placeandroute.problemgraph import parse_cnf
from multiprocessing import Pool
import dwave_networkx as dwnx

def cnf_to_constraints(clauses, num_vars):
    #from each 3-cnf clause, generate a Constraint with a fresh ancilla variable
    ancilla = num_vars + 1
    for clause in clauses:
        #                 first two vars        third var + ancilla
        yield Constraint([map(abs, clause[:2]),[abs(clause[2]), ancilla]])
        ancilla += 1


if __name__ == '__main__':
    #open 3-sat problem file
    with open(dirname(__file__) + "/../simple60.cnf") as f:
        cnf = (parse_cnf(f))
    cnf = [map(lambda x: x // 2, clause) for clause in cnf[:130]] #half vars, original is too big

    #prepare constraints
    constraints = list(cnf_to_constraints(cnf, max(max(x) for x in cnf)))

    #prepare tile graph
    chimera_size = 16
    tile_graph, choices = chimeratiles(chimera_size, chimera_size)

    #initialize and run heuristic
    heuristic = ParallelPlacementHeuristic(constraints, tile_graph, choices)
    pool = Pool()
    success = heuristic.par_run(pool)

    #print results
    if success:
        print_("Success")
        # constraint_placement is a map from constraint to tile
        for c, t in iteritems(heuristic.constraint_placement):
            print_(c.tile, t)

        print_("Expanding chains")
        # heuristic.chains maps from variable to tile, expand to a map variable->qubit
        chains = expand_solution(tile_graph, heuristic.chains, dwnx.chimera_graph(chimera_size))

        print_(repr(chains))
    else:
        print_("Failure")