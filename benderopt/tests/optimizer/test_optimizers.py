import numpy as np

from benderopt.minimizer import minimize
from benderopt.optimizer.random import RandomOptimizer

optimization_problem = [
    {"name": "x1", "category": "uniform", "search_space": {"low": 0, "high": 10}},
    {"name": "x1_step", "category": "uniform", "search_space": {"low": 0, "high": 10, "step": 1}},
    {"name": "x2", "category": "loguniform", "search_space": {"low": 1e4, "high": 1e6, "base": 10}},
    {
        "name": "x2_step",
        "category": "loguniform",
        "search_space": {"low": 1e4, "high": 1e6, "step": 1e3, "base": 10},
    },
    {
        "name": "x3",
        "category": "normal",
        "search_space": {"mu": 8, "sigma": 4, "low": 0, "high": 10},
    },
    {
        "name": "x3_step",
        "category": "normal",
        "search_space": {"mu": 8, "sigma": 4, "low": 0, "high": 10, "step": 0.2},
    },
    {
        "name": "x4",
        "category": "lognormal",
        "search_space": {"mu": 1e-5, "sigma": 1e1, "low": 1e-7, "high": 1e-3, "base": 10},
    },
    {
        "name": "x4_step",
        "category": "lognormal",
        "search_space": {
            "mu": 1e-5,
            "sigma": 1e1,
            "low": 1e-8,
            "high": 1e-3,
            "step": 1e-8,
            "base": 10,
        },
    },
    {"name": "x5", "category": "categorical", "search_space": {"values": ["a", "b", "c", "d"]}},
]


def function_to_optimize(x1, x1_step, x2, x2_step, x3, x3_step, x4, x4_step, x5):
    loss = 0
    loss += ((x1 - 5.5) / 5.5) ** 2
    loss += ((x1_step - 5) / 5) ** 2
    loss += ((x2 - 3.75e4) / 3.75e4) ** 2
    loss += ((x2_step - 9.2e4) / 9.2e4) ** 2
    loss += ((x3 - 8.1447) / 8.1447) ** 2
    loss += ((x3_step - 8) / 8) ** 2
    loss += ((x4 - 1.33e-5) / 1.33e-5) ** 2
    loss += ((x4_step - 1.456e-5) / 1.456e-5) ** 2
    loss += ["a", "b", "c", "d"].index(x5) / (3 + 2 + 1 + 0)
    return loss


best_sample = {
    "x1": 5.5,
    "x1_step": 5,
    "x2": 3.75e4,
    "x2_step": 9.2e4,
    "x3": 8.1447,
    "x3_step": 8,
    "x4": 1.33e-5,
    "x4_step": 1.456e-5,
    "x5": "a",
}


def test_random_uniform():
    minimize(
        function_to_optimize, optimization_problem, optimizer_type="random", number_of_evaluation=5
    )


def test_random_parzen_estimators():
    minimize(
        function_to_optimize,
        optimization_problem,
        optimizer_type="parzen_estimator",
        number_of_evaluation=35,
    )


def test_passing_optimizer_directly():
    minimize(
        function_to_optimize,
        optimization_problem,
        optimizer_type=RandomOptimizer,
        number_of_evaluation=5,
    )


def test_suggestions_with_global_seed_reset():
    suggestions = []

    def eval_(x: float) -> float:
        suggestions.append(x)
        np.random.seed(0)
        return 1.0

    params = [{"name": "x", "category": "uniform", "search_space": {"low": 0, "high": 1}}]
    minimize(eval_, params, optimizer_type="parzen_estimator", number_of_evaluation=5)

    assert len(suggestions) == len(set(suggestions))


# TODO: Add test for parallel minimize and exception handling
