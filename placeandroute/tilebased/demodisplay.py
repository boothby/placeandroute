from matplotlib import pyplot as plt
from placeandroute.tilebased.tactics import Tactic

class DoNothing(Tactic):
    def __str__(self):
        return "[do nothing]"

    def run(self):
        pass

class DrawIntermediateResult(Tactic):
    _global_initialization = False

    def __str__(self):
        return "[{} then display]".format(self._subtactic)

    def global_init(self):
        plt.ion()
        self._global_initialization = True

    def global_deinit(self):
        plt.show()
        self._global_initialization = False

    def __init__(self, placement, subTactic=DoNothing.default()):
        Tactic.__init__(self, placement)
        self._subtactic = subTactic.create(placement)
        if not self._global_initialization:
            self.global_init()


    def run(self):
        self.run()
        plt.show(block=False)

    def __del__(self):
        if self._global_initialization:
            self.global_deinit()

