"""Bayesian Optimization for In-Situ Optimization (Gaussian Process Based).

This module provides a function to optimize a user-supplied objective function
via Bayesian Optimization, using the `bayesian-optimization` library. The
function returns the best solution found and its performance after performing
an iterative Bayesian search.

Installation:
    pip install bayesian-optimization

Example:
    >>> import numpy as np
    >>> from my_bayes_opt import optimize_bayesian
    >>>
    >>> def user_objective(x):
    ...     # x is a NumPy array of parameters
    ...     return 40 - np.sum((x - 0.3) ** 2)
    >>>
    >>> best_sol, best_perf = optimize_bayesian(
    ...     objective_function=user_objective,
    ...     num_parameters=5,
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


def optimize_bayesian(
    objective_function,
    num_parameters,
    init_points=5,
    n_iter=25,
    init_min=-1.0,
    init_max=1.0
):
    """Optimize an objective function with Bayesian Optimization using a Gaussian Process.

    Args:
        objective_function (callable): A function f(x) -> float that takes a 1D NumPy array
            and returns a performance value (the higher, the better).
        num_parameters (int): Dimensionality of the parameter vector to optimize.
        init_points (int, optional): Number of random exploration points before
            beginning the Bayesian search.
        n_iter (int, optional): Number of Bayesian Optimization steps (model-based picks).
        init_min (float, optional): Lower bound for each parameter.
        init_max (float, optional): Upper bound for each parameter.

    Returns:
        tuple:
            - np.ndarray: The best solution (parameter vector) found.
            - float: The best performance value (objective function output).

    Notes:
        - Internally, each parameter is labeled 'param{i}', where i in [0..num_parameters-1].
        - The bayesian-optimization library constructs a Gaussian Process-based model
          to iteratively select new parameter vectors to evaluate.
    """
    # Build parameter bounds for the BayesianOptimization object
    pbounds = {
        f"param{i}": (init_min, init_max)
        for i in range(num_parameters)
    }

    def black_box_function(**kwargs):
        """Convert parameter dictionary to a NumPy vector and evaluate the objective function."""
        x = np.zeros(num_parameters, dtype=float)
        for i in range(num_parameters):
            x[i] = kwargs[f"param{i}"]
        return objective_function(x)

    # Initialize Bayesian Optimization
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,
        random_state=1
    )

    # Run optimization
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    # Retrieve best parameters
    best_params_dict = optimizer.max['params']
    best_solution = np.zeros(num_parameters, dtype=float)
    for i in range(num_parameters):
        best_solution[i] = best_params_dict[f"param{i}"]

    best_performance = optimizer.max['target']
    return best_solution, best_performance
