import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
jax.config.update('jax_enable_x64', True)

from dfbench import OptimizationAlgorithm
from relay import Relay


class N6Relay(OptimizationAlgorithm):
    algorithm_str = 'n6_relay'

    def __init__(self):
        pass

    def optimize(self, objective, init_params=None, random_seed=None, **kw):
        seed, _ = self.prepare(objective, unbounded=False, random_seed=random_seed)
        Relay(objective, seed, init_params).run()
