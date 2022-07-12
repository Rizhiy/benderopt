import logging
import multiprocessing as mp
import os
import time
import traceback
from typing import Any, Callable, Dict, List, Type, Union

import numpy as np
from tqdm import tqdm, trange

from benderopt.base import Observation, OptimizationProblem
from benderopt.optimizer import optimizers
from benderopt.optimizer.optimizer import BaseOptimizer
from benderopt.optimizer.parzen_estimator import ParzenEstimator
from benderopt.rng import RNG

LOGGER = logging.getLogger(__name__)


def _execute_func(f: Callable, sample: Dict[str, Any]) -> Any:
    try:
        result = f(**sample)
    except Exception:
        LOGGER.exception("Got an exception while evaluating {} on {}".format(f, sample))
        result = np.inf
    return result


def minimize(
    f,
    optimization_problem_parameters,
    optimizer_type="parzen_estimator",
    number_of_evaluation=100,
    seed=None,
    debug=False,
):
    logger = logging.getLogger("benderopt")

    RNG.seed(seed)

    optimization_problem = OptimizationProblem.from_list(optimization_problem_parameters)

    if isinstance(optimizer_type, str):
        optimizer_type = optimizers[optimizer_type]
    if not issubclass(optimizer_type, BaseOptimizer):
        raise ValueError(
            "optimizer_type should either be a string or a subclass of BaseOptimizer, got {}".format(
                optimizer_type
            )
        )
    optimizer = optimizer_type(optimization_problem)

    tbar = trange(number_of_evaluation, desc="Optimizing")
    best_loss = np.inf
    for iter_idx in tbar:
        logger.info("Evaluating {0}/{1}...".format(iter_idx + 1, number_of_evaluation))
        sample = optimizer.suggest()
        loss = _execute_func(f, sample)
        if loss < best_loss:
            tbar.set_description("Optimizing, best = {loss:.2e}".format(loss=loss))
            best_loss = loss
        logger.debug("f={0} for optimizer suggestion: {1}.".format(loss, sample))
        optimization_problem.add_observation(
            Observation.from_dict({"loss": loss, "sample": sample})
        )
    if debug:
        return optimization_problem.samples
    return optimization_problem.best_sample


def _worker(
    f: Callable, sample_queue: mp.Queue, result_queue: mp.Queue, seed=None, main_bar: tqdm = None
):
    # This is technically incorrect since all processes will use the same random sequence
    # TODO: Need to think about how to initialise it better
    if main_bar:
        tqdm._instances.add(main_bar)
    RNG.seed(seed)
    while True:
        sample = sample_queue.get()
        if sample is None:
            break
        result = _execute_func(f, sample)
        result_queue.put((sample, result))


def parallel_minimize(
    eval_func: Callable[..., float],
    problem: Union[OptimizationProblem, List[Dict[str, Any]]],
    optimizer: Union[Type[BaseOptimizer], str] = ParzenEstimator,
    num_runs: int = None,
    seed=None,
    return_all=False,
    num_proc: int = None,
) -> Union[Observation, List[Observation]]:
    """
    Run multiple evaluations in parallel. Should be much faster on single-threaded tasks
    Performance might be decreased because of overlaps, increase num_runs to compensate.

    :param eval_func: Function to optimise
    :param problem: Problem to optimise or list of parameters
    :param optimizer: Which optimizer to use
    :param num_runs: How many evaluations to perform
    :param seed: Seed for random number generator
    :param return_all: Whether to return all observations or just the best
    :param num_proc: How many processes to use
    """
    logger = logging.getLogger("benderopt.parallel_minimize")

    if not isinstance(problem, OptimizationProblem):
        problem = OptimizationProblem.from_list(problem)
    if not problem.parameters:
        raise ValueError("No problem_parameters have been provided")

    num_params = len(problem.parameters)
    num_runs = num_runs or len(num_params) ** 3  # Empirically seems to produce good results
    if num_runs < num_params**2:
        logger.warning(
            f"{num_runs} is usually not enough to properly optimise {num_params} parameters"
        )

    num_proc = min(num_proc or os.cpu_count(), num_runs)

    RNG.seed(seed)

    if isinstance(optimizer, str):
        optimizer = optimizers[optimizer]
    if not issubclass(optimizer, BaseOptimizer):
        raise ValueError(
            f"optimizer_type should either be a string or a subclass of BaseOptimizer, got {optimizer}"
        )
    optimizer = optimizer(problem)

    sample_queue = mp.Queue()
    result_queue = mp.Queue()
    tbar = trange(num_runs, desc="Optimizing")

    proc_args = eval_func, sample_queue, result_queue, seed, tbar
    processes = [mp.Process(target=_worker, args=proc_args) for _ in range(num_proc)]
    for proc in processes:
        proc.start()
        sample_queue.put(optimizer.suggest())

    best_loss = np.inf
    for iter_idx in tbar:
        sample, loss = result_queue.get()
        logger.debug(f"loss={loss:.2e} for optimizer suggestion: {sample}")
        if loss < best_loss:
            tbar.set_description(f"Optimizing, best = {loss:.2e}")
            best_loss = loss
        problem.add_observation(Observation.from_dict({"loss": loss, "sample": sample}))
        sample_queue.put(optimizer.suggest() if iter_idx < num_runs - num_proc else None)

    sleep_time = 0.01
    sleeps_remaining = int(1 / sleep_time)
    for proc in processes:
        proc.terminate()
        while sleeps_remaining:  # Wait at most 1 sec
            if not proc.is_alive():
                break
            time.sleep(sleep_time)
            sleeps_remaining -= 1
        proc.kill()
        proc.close()

    obss = problem.sorted_observations
    return obss if return_all else obss[0]


if __name__ == "__main__":

    def f(x):
        return np.sin(x)

    optimization_problem_parameters = [
        {"name": "x", "category": "uniform", "search_space": {"low": 0, "high": 2 * np.pi}}
    ]

    best_sample = minimize(
        f, optimization_problem_parameters=optimization_problem_parameters, number_of_evaluation=100
    )

    print(best_sample["x"], 3 * np.pi / 2)
