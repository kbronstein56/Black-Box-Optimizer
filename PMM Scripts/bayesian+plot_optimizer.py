"""
Bayesian Optimization with Detailed Plotting (1D Only).

This module provides a function to optimize a one-dimensional objective function
via Bayesian Optimization using a Gaussian Process. In addition to performing the
optimization, it produces a plot showing:
  - Observed objective values (blue dots)
  - The true objective function (dotted black line)
  - The GP surrogate mean (solid red line)
  - The 95% confidence interval of the GP (shaded red region)

Installation:
    pip install bayesian-optimization matplotlib

Example:
    >>> import numpy as np
    >>> from bayesian_plot_optimizer import optimize_bayesian_with_detailed_plot, test_objective
    >>> best_sol, best_perf = optimize_bayesian_with_detailed_plot(
    ...     objective_function=test_objective,
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
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization


def optimize_bayesian_with_detailed_plot(
    objective_function,
    num_parameters,
    init_points=5,
    n_iter=25,
    init_min=-1.0,
    init_max=1.0
):
    """Optimize a one-dimensional objective function using Bayesian Optimization and plot detailed results.

    This function is designed for a one-dimensional parameter space (num_parameters must be 1).
    It uses the bayesian-optimization library to build a Gaussian Process surrogate and then
    produces a plot showing observed data, the true function, the GP mean, and a 95% confidence interval.

    Args:
        objective_function (callable): A function f(x) -> float that takes a 1D NumPy array
            and returns a performance value (the higher, the better).
        num_parameters (int): Dimensionality of the parameter vector (must be 1 for detailed plotting).
        init_points (int, optional): Number of random exploration points.
        n_iter (int, optional): Number of Bayesian Optimization iterations.
        init_min (float, optional): Lower bound for the parameter.
        init_max (float, optional): Upper bound for the parameter.

    Returns:
        tuple:
            - np.ndarray: The best solution (parameter vector) found.
            - float: The best performance value.
    """
    if num_parameters != 1:
        raise ValueError("This detailed plotting version only supports num_parameters == 1.")

    # Define parameter bounds.
    pbounds = {"param0": (init_min, init_max)}

    def black_box_function(**kwargs):
        """Convert dict to array and evaluate the objective function."""
        x = np.array([kwargs["param0"]], dtype=float)
        return objective_function(x)

    # Initialize Bayesian Optimization.
    optimizer = BayesianOptimization(
        f=black_box_function,
        pbounds=pbounds,
        verbose=2,
        random_state=1
    )

    # Run the optimization.
    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    # Retrieve best parameters.
    best_params_dict = optimizer.max["params"]
    best_solution = np.array([best_params_dict["param0"]], dtype=float)
    best_performance = optimizer.max["target"]

    # For detailed plotting, create a grid over the parameter range.
    grid = np.linspace(init_min, init_max, 200).reshape(-1, 1)
    # Predict using the internal GP surrogate.
    gp_mean, gp_std = optimizer._gp.predict(grid, return_std=True)
    gp_upper = gp_mean + 1.96 * gp_std
    gp_lower = gp_mean - 1.96 * gp_std

    # Get observed data points from optimization history.
    observed_x = np.array([res["params"]["param0"] for res in optimizer.res])
    observed_y = np.array([res["target"] for res in optimizer.res])

    # For demonstration, plot the true function if available.
    # Here we assume objective_function is test_objective, so we compute true values.
    true_y = np.array([objective_function(np.array([x])) for x in grid.flatten()])

    # Create the plot.
    plt.figure(figsize=(10, 6))
    plt.plot(grid, true_y, 'k--', label="True Function")       # Dotted black line.
    plt.plot(grid, gp_mean, 'r-', label="GP Mean")               # Solid red line.
    plt.fill_between(grid.flatten(), gp_lower, gp_upper, color='r', alpha=0.2, label="95% Confidence")  # Shaded area.
    plt.scatter(observed_x, observed_y, c='b', s=40, label="Observed Points")  # Blue dots.
    plt.xlabel("Parameter Value")
    plt.ylabel("Objective Performance")
    plt.title("Bayesian Optimization Progress")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"[Bayesian Optimization] Best performance achieved: {best_performance:.4f}")
    print(f"[Bayesian Optimization] Best solution (parameter vector): {best_solution}")

    return best_solution, best_performance


def test_objective(x: np.ndarray) -> float:
    """Test objective function for demonstration.

    This function computes:
        performance = 40 - sum((x - 0.3)^2)
    The optimal performance of 40 is achieved when x == [0.3].

    Args:
        x (np.ndarray): A NumPy array containing the parameter.

    Returns:
        float: The performance value.
    """
    ideal_value = 0.3
    error = np.sum((x - ideal_value) ** 2)
    performance = 40 - error
    return performance


if __name__ == "__main__":
    # Run a test when executed as a script.
    best_sol, best_perf = optimize_bayesian_with_detailed_plot(
        objective_function=test_objective,
        num_parameters=1,
        init_points=5,
        n_iter=10,
        init_min=-1.0,
        init_max=1.0
    )
    print("Test run complete.")
    print("Best solution:", best_sol)
    print("Best performance:", best_perf)
