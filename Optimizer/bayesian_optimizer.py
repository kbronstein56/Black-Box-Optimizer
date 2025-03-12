"""Bayesian Optimization for In-Situ Optimization (Gaussian Process Based).

This module provides a function to optimize a user-supplied objective function
via Bayesian Optimization, using the `bayesian-optimization` library. 
Installation:
    pip install bayesian-optimization matplotlib

Example:
    >>> import numpy as np
    >>> from bayesian_optimizer import optimize_bayesian
    >>>
    >>> def user_objective(x):
    ...     # x is a 1D NumPy array
    ...     return 40 - np.sum((x - 0.3)**2)
    >>>
    >>> best_sol, best_perf = optimize_bayesian(
    ...     objective_function=user_objective,
    ...     num_parameters=1,
    ...     init_points=5,
    ...     n_iter=10,
    ...     init_min=-1.0,
    ...     init_max=1.0
    ... )
    >>> print("Best solution:", best_sol)
    >>> print("Best performance:", best_perf)
"""

import numpy as np
from bayes_opt import BayesianOptimization
import time

def optimize_bayesian(
    objective_function,
    num_parameters,
    init_points=5,
    n_iter=25,
    init_min=-1.0,
    init_max=1.0
):
    """Optimize an objective function with Bayesian Optimization using a Gaussian Process.

    This function builds a parameter bounds dictionary, then performs an initial
    exploration phase with a given number of random points. Afterwards, it iteratively
    runs one Bayesian optimization step at a time and prints the generation and the current
    best performance in each iteration.

    Args:
        objective_function (callable): A function f(x) -> float that takes a 1D NumPy array
            and returns a performance value (the higher, the better).
        num_parameters (int): Dimensionality of the parameter vector to optimize.
        init_points (int, optional): Number of random exploration points before
            beginning the Bayesian search.
        n_iter (int, optional): Number of Bayesian Optimization iterations (model-based picks).
        init_min (float, optional): Lower bound for each parameter.
        init_max (float, optional): Upper bound for each parameter.

    Returns:
        tuple:
            - np.ndarray: The best solution (parameter vector) found.
            - float: The best performance value (objective function output).

    Notes:
        - Each parameter is internally labeled 'param{i}' for i in [0, num_parameters-1].
    """
    # Build parameter bounds for the BayesianOptimization object.
    pbounds = {f"param{i}": (init_min, init_max) for i in range(num_parameters)}

    def black_box_function(**kwargs):
        """Convert parameter dictionary to a NumPy vector and evaluate the objective function."""
        x = np.zeros(num_parameters, dtype=float)
        for i in range(num_parameters):
            x[i] = kwargs[f"param{i}"]
        return objective_function(x)

    # Initialize Bayesian Optimization.
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=0,  # We control our own printing.
        random_state=1
    )

    # initial random exploration.
    optimizer.maximize(init_points=init_points, n_iter=0)

    for gen in range(n_iter):
        optimizer.maximize(init_points=0, n_iter=1)
        current_best = optimizer.max['target']
        print(f"[BO] Generation {gen+1}/{n_iter}, Best Performance: {current_best:.4f}")

    best_params_dict = optimizer.max['params']
    best_solution = np.zeros(num_parameters, dtype=float)
    for i in range(num_parameters):
        best_solution[i] = best_params_dict[f"param{i}"]

    best_performance = optimizer.max['target']
    return best_solution, best_performance

if __name__ == "__main__":
    # test when run as a script.
    def test_objective(x):
        return 40 - np.sum((x - 0.3)**2)
    best_sol, best_perf = optimize_bayesian(test_objective, num_parameters=1, init_points=5, n_iter=10)
    print("Best solution:", best_sol)
    print("Best performance:", best_perf)
