"""Standard CMA-ES Optimizer

Provides a simplified implementation of the Covariance Matrix Adaptation
Evolution Strategy (CMA-ES). The algorithm maintains a mean vector (representing
the current best guess of the solution) and an overall step size (sigma) for sampling.
It updates the mean and step size each iteration based on the performance of sampled
solutions.

Returns the best solution found and its performance.

Example:
    >>> def objective_function(x):
    ...     return -sum((x - 0.5)**2)
    >>> best_sol, best_perf = optimize_cma_es(objective_function, num_parameters=10)
    >>> print(best_sol, best_perf)
"""

import math
import numpy as np


def optimize_cma_es(
    objective_function,
    num_parameters,
    init_sigma=0.3,
    lower_bound=-1.0,
    upper_bound=1.0,
    max_iter=50
):
    """Optimize a user-supplied objective function using a simplified CMA-ES approach.

    This version of CMA-ES updates:
      - A mean vector, sampled from within [lower_bound, upper_bound]
      - An overall step size (sigma)
    It ranks the sampled individuals, and uses a weighted average of the top half
    to shift the mean. Step size is updated via the standard deviation of the solutions.

    Args:
        objective_function (callable): A function f(x) -> float that evaluates
            the performance of a parameter vector (higher is better).
        num_parameters (int): Dimensionality of the parameter space.
        init_sigma (float, optional): Initial step size used for sampling.
        lower_bound (float, optional): Lower bound for each parameter.
        upper_bound (float, optional): Upper bound for each parameter.
        max_iter (int, optional): Number of iterations to run CMA-ES.

    Returns:
        tuple:
            - np.ndarray: The best solution (parameter vector) found.
            - float: The best performance (fitness) achieved by that solution.
    """
    # Initialize the mean vector randomly within the bounds
    mean_vector = np.random.uniform(lower_bound, upper_bound, num_parameters)
    sigma = init_sigma
    population_size = 4 + int(3 * math.log(num_parameters))

    # Evaluate initial mean
    best_solution = mean_vector.copy()
    best_performance = objective_function(mean_vector)

    for iteration in range(max_iter):
        # Sample a population around the current mean with step size sigma
        solutions = []
        performances = []
        for _ in range(population_size):
            candidate = mean_vector + sigma * np.random.randn(num_parameters)
            candidate = np.clip(candidate, lower_bound, upper_bound)
            solutions.append(candidate)

            perf = objective_function(candidate)
            performances.append(perf)

            # Track best solution overall
            if perf > best_performance:
                best_performance = perf
                best_solution = candidate.copy()

        # Sort the population by performance in descending order
        indices = np.argsort(performances)[::-1]
        top_count = population_size // 2

        # Compute weights for top individuals
        weights = np.array([top_count - i for i in range(top_count)], dtype=float)
        weights /= np.sum(weights)

        # Weighted average of the top solutions to get new mean
        new_mean = np.zeros(num_parameters)
        for i in range(top_count):
            new_mean += weights[i] * solutions[indices[i]]

        mean_vector = new_mean

        # Update sigma (overall step size) based on population spread
        sigma = 0.9 * sigma + 0.1 * np.std(solutions, axis=0).mean()

        print(
            f"[CMA-ES] Iteration {iteration + 1}/{max_iter}, "
            f"Best Performance: {best_performance:.4f}"
        )

    return best_solution, best_performance
