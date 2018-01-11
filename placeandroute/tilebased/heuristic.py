import random

from placeandroute.routing.bonnheuristic import MinMaxRouter


class TilePlacementHeuristic(object):
    def __init__(self, constraints, archGraph):
        # list of constraints. each constraint has a set of variables and a weight. could use the factor graph
        self.constraints = constraints
        # target graph. each node is a set of edges in the original graph, each edge is a set of frontier vertices
        # somehow the possible choices are tagged
        # xxx: edges can be grouped if they are symmetrical, verify this
        self.arch = archGraph
        self.rounds = 1000
        self.placement = dict()

    def init_placement_random(self):
        choices = self.arch.nodes()
        for constraint in self.constraints:
            possible_choices = [x for x in choices if self.arch.node[x]['capacity'] >= constraint.weight]
            if not possible_choices:
                return False
            position = random.choice(possible_choices)
            self.placement[constraint] = position
            self.arch.node[position]['capacity'] -= constraint.weight
        self.do_routing()
        return True

    def do_routing(self):
        placement = None
        router = MinMaxRouter(self.arch, placement)
        router.run(self.rounds)
        #todo
        pass


    def run(self):
        if not self.init_placement_random():
            return False


        for round in xrange(self.rounds):
            if self.is_valid_embedding():
                return True
            remove_choice = self.pick_worst()
            self.remove_tile(clean_chains=True)
            best_place = self.find_best_place(remove_choice)
            self.insert_tile(remove_choice, best_place)

        return False

    def is_valid_embedding(self):
        pass

    def pick_worst(self):
        pass

    def remove_tile(self, clean_chains=True):
        pass

    def find_best_place(self, constraint):
        pass

    def insert_tile(self, constraint, tile):
        pass
