from multiprocessing import Pool

from typing import Optional

from placeandroute.tilebased.heuristic import TilePlacementHeuristic


class ParallelPlacementHeuristic(TilePlacementHeuristic):
    """Heuristic constraint placer. Tries to decrease overused qubits by ripping and rerouting"""

    def par_run(self, pool, stop_first=False):
        # type: (Pool, Optional[bool]) -> bool
        """Run heuristic in a process pool, pick result according to stop_first

        todo: use context interface for pool
        """
        ret = False
        if stop_first:
            for subret, subobj in pool.imap_unordered(self._run_stop, self._init_tactics):
                if subret:
                    ret = True
                    self.chains = subobj.chains
                    self.constraint_placement = subobj.constraint_placement
                    break
            pool.terminate()  # stop pending searches
        else:
            for subret, subobj in pool.imap_unordered(self._run_full, self._init_tactics):
                ret = ret or subret
                self.chains = subobj.chains
                self.constraint_placement = subobj.constraint_placement
                self.save_best()
            self.restore_best()
        return ret

    def _run_stop(self, init_tat):
        subobj = TilePlacementHeuristic(self.constraints, self.arch, self.choices, init_tactics=[init_tat])
        return subobj.run(stop_first=True), subobj

    def _run_full(self, init_tat):
        subobj = TilePlacementHeuristic(self.constraints, self.arch, self.choices, init_tactics=[init_tat])
        return subobj.run(stop_first=False), subobj
