import numpy as np

from benderopt.base import OptimizationProblem
from benderopt.optimizer import optimizers


def test_base_optimize_ok():

    optimization_problem = [
        {"name": "x", "category": "uniform", "search_space": {"low": 0, "high": np.pi}}
    ]

    optimization_problem = OptimizationProblem.from_list(optimization_problem)
    optimizer = optimizers["random"](optimization_problem)
    assert optimizer.observations == optimization_problem.observations
    optimizer = optimizers["random"](optimization_problem)
    assert len(optimizer.suggest(10)) == 10
