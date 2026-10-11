import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dfbench import OptimizationAlgorithm
from portfolio import Portfolio


class ChampionRepair(OptimizationAlgorithm):
    algorithm_str = 'champion_repair'

    def __init__(self):
        pass

    def optimize(self, objective, init_params=None, random_seed=None, **kw):
        seed, _ = self.prepare(objective, unbounded=True, random_seed=random_seed)
        Portfolio(objective, seed, 'repair', init_params).run()
