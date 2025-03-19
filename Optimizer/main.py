"""Main Script for In-Situ Optimization Package.

This script interactively asks questions about your experiment and your objective function.
Based on your answers (number of parameters, evaluation budget, noise level, smoothness, etc.),
the script automatically suggests an optimizer from the following list:

  - GA            : Genetic Algorithm
  - CMA-ES        : Standard Covariance Matrix Adaptation Evolution Strategy
  - CMA-ES-GI     : CMA-ES with Gradient Information
  - Bayesian      : Bayesian Optimization (Gaussian Process based)
  - ActorCritic   : Actor-Critic Reinforcement Learning

It then asks if you wish to proceed with the suggested optimizer or choose another.
You must also decide whether to use the example objective function or provide your own
(by supplying a module that defines a function called 'user_objective').

Example objective function:

    def user_objective(x):
        # x is a NumPy array of parameters
        # Optimal performance of 40 is achieved when each parameter equals 0.3
        return 40 - np.sum((x - 0.3)**2)

After the optimizer is run, the script prints the best solution and performance along with the total runtime.
"""

import sys
import importlib
import time
import numpy as np

# Import the separate optimizer modules
from Optimizer.ga_optimizer import optimize_ga
from Optimizer.cma_es_optimizer import optimize_cma_es
from Optimizer.cma_es_gi_optimizer import optimize_cma_es_gi
from Optimizer.bayesian_optimizer import optimize_bayesian
from Optimizer.actor_critic_optimizer import optimize_actor_critic


def example_objective_function(parameters: np.ndarray) -> float:
    """An example objective function used for demonstration.

    This function returns:
        performance = 40 - sum((parameters - 0.3)^2)

    The optimal performance (40) is achieved when every parameter equals 0.3.

    Args:
        parameters (np.ndarray): A NumPy array of shape (num_params,).

    Returns:
        float: The performance value, higher is better.
    """
    ideal_value = 0.3
    error = np.sum((parameters - ideal_value) ** 2)
    performance = 40 - error
    return performance


def main():
    """Run an interactive session to select and execute an optimization algorithm.

    This function asks the user:
      1. The number of parameters in their experiment.
      2. The approximate evaluation budget (how many times the objective can be called).
      3. The noise level (high, moderate, low).
      4. Whether the function is smooth and continuous.

    Based on answers, it suggests one of the following optimizers:
      - GA
      - CMA-ES
      - CMA-ES-GI
      - Bayesian
      - ActorCritic

    The suggestion now also considers ActorCritic if the evaluation budget is high and the user wants a temporally responsive device.
    The user can accept or override the suggestion. Then the user decides if they want to provide a custom objective
    function or use the included example objective. Finally, the chosen optimizer is run, and the best solution and
    performance are printed.
    """
    print("Welcome to the In-Situ Optimization Package!")
    print("Supported optimizers:")
    print("  GA            : Genetic Algorithm")
    print("  CMA-ES        : Standard CMA-ES")
    print("  CMA-ES-GI     : CMA-ES with Gradient Information")
    print("  Bayesian      : Bayesian Optimization")
    print("  ActorCritic   : Actor-Critic Reinforcement Learning")
    print()

    # ------------------------------------------------------------
    # Ask user for essential inputs: number of parameters, evaluations, etc.
    # ------------------------------------------------------------
    try:
        num_params = int(input("Enter the number of parameters in your experiment (e.g., 100, 200): "))
    except ValueError:
        print("Invalid input; defaulting to 100 parameters.")
        num_params = 100

    try:
        eval_budget = int(input("Approximately, how many function evaluations can your experiment support? (e.g., 50, 1000): "))
    except ValueError:
        print("Invalid input; defaulting to 100 evaluations.")
        eval_budget = 100

    noise_level = input("What is the noise level of your objective function? (high, moderate, low): ").strip().lower()
    smooth_response = input("Is your objective function smooth and continuous? (yes/no): ").strip().lower()
    temporally_responsive = input("Do you need real-time or adaptive optimization (i.e., many evaluations over time)? (yes/no): ").strip().lower()

    # ------------------------------------------------------------
    # Determine suggested optimizer
    # ------------------------------------------------------------
    suggested_optimizer = None
    explanation = ""

    if eval_budget < 50:
        suggested_optimizer = "Bayesian"
        explanation += (
            "Your evaluation budget is low. Bayesian Optimization builds a probabilistic "
            "surrogate model (typically a Gaussian Process) to predict performance and "
            "select promising points, thereby minimizing the number of expensive function "
            "evaluations.\n"
        )
    elif noise_level == "high":
        suggested_optimizer = "GA"
        explanation += (
            "Your objective function is reported as high in noise. Genetic Algorithms use "
            "populations and stochastic operators (selection, crossover, mutation) that "
            "tend to be robust in noisy environments.\n"
        )
    elif num_params >= 100 and smooth_response == "yes":
        if temporally_responsive == "yes":
            suggested_optimizer = "ActorCritic"
            explanation += (
                "For a high-dimensional, smooth objective function with a need for real-time adaptation, "
                "the Actor-Critic approach is suitable because it can continuously learn and adjust over many "
                "episodes.\n"
            )
        else:
            suggested_optimizer = "CMA-ES-GI"
            explanation += (
                "With a high-dimensional parameter space and a smooth objective function, "
                "CMA-ES with Gradient Information can efficiently use partial gradient signals "
                "to accelerate convergence.\n"
            )
    else:
        suggested_optimizer = "CMA-ES"
        explanation += (
            "Given your inputs, a standard CMA-ES is a good choice as it adapts the "
            "covariance matrix to shape the search distribution in continuous spaces.\n"
        )

    print("\nBased on your answers:")
    print(f"  - Number of parameters: {num_params}")
    print(f"  - Evaluation budget: {eval_budget}")
    print(f"  - Noise level: {noise_level}")
    print(f"  - Smoothness: {smooth_response}")
    print(f"  - Real-time adaptation needed: {temporally_responsive}")
    print(f"\nSuggested Optimizer: {suggested_optimizer}")
    print("Reasoning:")
    print(explanation)

    # ------------------------------------------------------------
    # Ask user if they want to proceed with suggested optimizer
    # ------------------------------------------------------------
    user_response = input("Would you like to proceed with the suggested optimizer? (Y/N): ").strip().upper()
    if user_response != "Y":
        print("Available optimizers: GA, CMA-ES, CMA-ES-GI, Bayesian, ActorCritic")
        chosen_optimizer = input("Please enter your desired optimizer: ").strip()
    else:
        chosen_optimizer = suggested_optimizer

    # ------------------------------------------------------------
    # Ask about objective function
    # ------------------------------------------------------------
    print("\nDo you want to use the example objective function?")
    print("Example: def user_objective(x): return 40 - np.sum((x - 0.3)**2)")
    use_example = input("Enter Y for yes or N for no: ").strip().upper()

    if use_example == "Y":
        objective_function = example_objective_function
        print("\nUsing the example objective function.")
    else:
        print(
            "\nPlease provide the name of your Python module (without the .py extension) "
            "that defines a function named 'user_objective'. "
            "It should be located in the 'functions' folder."
        )
        module_name = input("Module name (e.g., my_objective): ").strip()
        try:
            user_module = importlib.import_module("Functions." + module_name)
            objective_function = user_module.user_objective
        except Exception as e:
            print("Error importing your module:", e)
            sys.exit(1)

    # ------------------------------------------------------------
    # Ask for custom parameter bounds
    # ------------------------------------------------------------
    print("\nWould you like to set custom parameter bounds? (default is [-1.0, 1.0])")
    custom_bounds = input("Enter Y for yes or N for no: ").strip().upper()
    if custom_bounds == "Y":
        try:
            lower_bound = float(input("Enter the lower bound for parameters (e.g., 40.0): "))
            upper_bound = float(input("Enter the upper bound for parameters (e.g., 60.0): "))
        except ValueError:
            print("Invalid input; using default bounds of -1.0 to 1.0.")
            lower_bound = -1.0
            upper_bound = 1.0
    else:
        lower_bound = -1.0
        upper_bound = 1.0

    print(f"Using parameter bounds: lower = {lower_bound}, upper = {upper_bound}")

    # ------------------------------------------------------------
    # Run the chosen optimizer
    # ------------------------------------------------------------
    print(f"\nRunning {chosen_optimizer.upper()} optimizer with {num_params} parameters...")
    start_time_opt = time.time()

    best_solution = None
    best_performance = -np.inf

    if chosen_optimizer.lower() == "ga":
        best_solution, best_performance = optimize_ga(objective_function, num_params,
                                                      init_min=lower_bound, init_max=upper_bound)
    elif chosen_optimizer.lower() == "cma-es":
        best_solution, best_performance = optimize_cma_es(objective_function, num_params,
                                                          lower_bound=lower_bound, upper_bound=upper_bound)
    elif chosen_optimizer.lower() == "cma-es-gi":
        best_solution, best_performance = optimize_cma_es_gi(objective_function, num_params,
                                                             lower_bound=lower_bound, upper_bound=upper_bound)
    elif chosen_optimizer.lower() == "bayesian":
        best_solution, best_performance = optimize_bayesian(objective_function, num_params,
                                                            init_min=lower_bound, init_max=upper_bound)
    elif chosen_optimizer.lower() == "actorcritic":
        best_solution, best_performance = optimize_actor_critic(objective_function, num_params,
                                                                lower_bound=lower_bound, upper_bound=upper_bound)
    else:
        print("Unknown optimizer choice. Exiting.")
        sys.exit(1)

    total_runtime = time.time() - start_time_opt

    # ------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------
    print("\nOptimization complete!")
    print(f"Best performance achieved: {best_performance:.4f}")
    print("Best solution (parameter vector):")
    print(best_solution)
    print(f"Total runtime: {total_runtime:.2f} seconds")


if __name__ == "__main__":
    main()
