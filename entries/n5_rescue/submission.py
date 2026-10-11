import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dfbench import OptimizationAlgorithm
from portfolio import Portfolio


class N5Rescue(OptimizationAlgorithm):
    algorithm_str = 'n5_rescue'

    def __init__(self):
        pass

    def optimize(self, objective, init_params=None, random_seed=None, **kw):
        seed, _ = self.prepare(objective, unbounded=True, random_seed=random_seed)
        Portfolio(objective, seed, 'rescue', init_params).run()
