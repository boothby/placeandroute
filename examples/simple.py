from os.path import dirname

from six import iteritems, print_

from placeandroute.tilebased.parallel import ParallelPlacementHeuristic
from placeandroute.tilebased.chimera_tiles import chimera_tiles as chimeratiles, expand_solution as expand_solution
import logging
from placeandroute.problemgraph import parse_cnf
from multiprocessing import Pool
import dwave_networkx as dwnx

from placeandroute.tilebased.utils import cnf_to_constraints, show_result

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s %(processName)s %(message)s')

    #open 3-sat problem file
    with open(dirname(__file__) + "/../simple60.cnf") as f:
        cnf = (parse_cnf(f))
    cnf = [list(map(lambda x: x // 2, clause)) for clause in cnf[:130]] #half vars, original is too big

    #prepare constraints
    constraints = list(cnf_to_constraints(cnf, max(max(x) for x in cnf)))

    #prepare tile graph
    chimera_size = 16
    tile_graph, choices = chimeratiles(chimera_size, chimera_size)

    #initialize and run heuristic
    heuristic = ParallelPlacementHeuristic(constraints, tile_graph, choices)
    pool = Pool()
    success = heuristic.par_run(pool, stop_first=True)
    #success = heuristic.run(stop_first=False)


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

        from matplotlib import pyplot as plt
        dwnx.draw_chimera_embedding(dwnx.chimera_graph(chimera_size), chains, node_size=50)
        plt.savefig('result.png')
    else:
        print_("Failure")
