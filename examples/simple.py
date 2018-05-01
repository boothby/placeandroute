from os.path import dirname

from placeandroute.tilebased.heuristic import Constraint, TilePlacementHeuristic
from placeandroute.tilebased.chimera_tiles import chimeratiles, expand_solution
from placeandroute.problemgraph import parse_cnf
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
    heuristic = TilePlacementHeuristic(constraints, tile_graph, choices)
    success = heuristic.run()

    #print results
    if success:
        print "Success"
        # constraint_placement is a map from constraint to tile
        for c, t in heuristic.constraint_placement.iteritems():
            print c.tile, t

        print "Expanding chains"
        # heuristic.chains maps from variable to tile, expand to a map variable->qubit
        chains = expand_solution(tile_graph, heuristic.chains, dwnx.chimera_graph(chimera_size))

        print repr(chains)
    else:
        print "Failure"