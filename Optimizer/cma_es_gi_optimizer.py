"""CMA-ES with Gradient Information Optimizer

This module provides a simplified Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
that incorporates a basic "Gradient Information" (GI) update. The GI step uses a rough
gradient approximation (based on differences in performance relative to the current mean)
to accelerate adaptation of the mean vector.

Returns the best solution found and its performance.

Example:
    >>> def objective_fn(x):
    ...     return 40 - sum((x - 0.3)**2)
    >>> best_sol, best_perf = optimize_cma_es_gi(objective_fn, num_parameters=10)
    >>> print(best_sol, best_perf)
"""

import math
import numpy as np


def optimize_cma_es_gi(
    objective_function,
    num_parameters,
    init_sigma=0.3,
    lower_bound=-1.0,
    upper_bound=1.0,
    max_iter=50
):
    """Optimize a user-supplied objective function using a CMA-ES variant that includes basic gradient information (GI).

    This variant:
      1. Maintains a mean vector for sampling.
      2. Uses a simple 'radius' (akin to sigma) to scale random vectors.
      3. Ranks the population to produce a rank-based mean shift.
      4. Computes a gradient estimate from performance differences relative to the mean.
      5. Blends that gradient into the mean update.

    Args:
        objective_function (callable): A function f(x) -> float that gives
            the performance for parameter vector x (higher is better).
        num_parameters (int): Dimensionality of the parameter vector.
        init_sigma (float, optional): Initial sampling radius (like sigma).
        lower_bound (float, optional): Minimum value for each parameter.
        upper_bound (float, optional): Maximum value for each parameter.
        max_iter (int, optional): Number of CMA-ES-GI iterations to run.

    Returns:
        tuple:
            - np.ndarray: The best parameter vector found.
            - float: The highest performance achieved.
    """
    # Initialize the mean vector and radius
    mean_vector = np.random.uniform(lower_bound, upper_bound, num_parameters)
    radius = init_sigma
    population_size = 4 + int(3 * math.log(num_parameters))

    # Evaluate initial mean
    best_solution = mean_vector.copy()
    best_performance = objective_function(mean_vector)

    # Main loop
    for iteration in range(max_iter):
        population = []
        performances = []

        # Sample population
        for _ in range(population_size):
            z = np.random.randn(num_parameters)
            candidate = mean_vector + radius * z
            candidate = np.clip(candidate, lower_bound, upper_bound)
            population.append(candidate)

            perf = objective_function(candidate)
            performances.append(perf)
            # Track best solution
            if perf > best_performance:
                best_performance = perf
                best_solution = candidate.copy()

        # Rank-based mean update
        indices = np.argsort(performances)[::-1]
        mu = population_size // 2
        weights = np.array([mu - i for i in range(mu)], dtype=float)
        weights = weights / np.sum(weights)
        new_mean = np.zeros(num_parameters)
        for i in range(mu):
            new_mean += weights[i] * population[indices[i]]

        # Compute approximate gradient using performance relative to current mean
        f_mean = objective_function(mean_vector)
        gradient = np.zeros(num_parameters)
        for candidate, perf in zip(population, performances):
            gradient += (perf - f_mean) * (candidate - mean_vector)
        gradient /= population_size

        # Blend gradient into the mean update
        new_mean += 0.1 * gradient
        mean_vector = new_mean

        # Adapt radius (akin to step size) based on population spread
        radius = 0.9 * radius + 0.1 * np.std(population, axis=0).mean()

        print(
            f"[CMA-ES-GI] Iteration {iteration + 1}/{max_iter}, "
            f"Best Performance: {best_performance:.4f}"
        )

    return best_solution, best_performance
