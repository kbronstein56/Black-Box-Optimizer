"""Genetic Algorithm Optimizer for In-Situ Optimization.

This module provides a function to optimize a user-supplied objective function
using a Genetic Algorithm (GA). Each individual is a vector of parameters,
and the GA employs:
  - Tournament selection
  - Blend crossover
  - Gaussian mutation

The function returns the best solution found and its corresponding performance
value (fitness).

Example:
    >>> def objective_function(x):
    ...     return -sum((x - 0.3)**2)  # A toy objective
    >>> best_sol, best_perf = optimize_ga(objective_function, 10)
    >>> print(best_sol, best_perf)
"""

import math
import random
import numpy as np


def optimize_ga(
    objective_function,
    num_parameters,
    population_size=100,
    max_generations=50,
    init_min=-1.0,
    init_max=1.0,
    crossover_rate=0.9,
    mutation_rate=0.1
):
    """Optimize a given objective function using a Genetic Algorithm (GA).

    This function initializes a population of individuals (parameter vectors),
    evolves them over several generations, and returns the best solution found
    along with its performance.

    Args:
        objective_function (callable): A function f(x) -> float that evaluates
            the performance of a parameter vector. Higher is better.
        num_parameters (int): The dimensionality of the parameter space.
        population_size (int): Number of individuals in the GA population.
        max_generations (int): How many generations to run the GA for.
        init_min (float): Lower bound for initializing parameters.
        init_max (float): Upper bound for initializing parameters.
        crossover_rate (float): Probability that crossover is applied.
        mutation_rate (float): Probability of mutating each gene.

    Returns:
        tuple:
            - np.ndarray: The best solution (parameter vector) found.
            - float: The performance (fitness) of that best solution.
    """

    def create_individual():
        """Create a random individual (vector) within the init_min, init_max range."""
        return np.array([
            random.uniform(init_min, init_max) for _ in range(num_parameters)
        ])

    def crossover(parent1, parent2):
        """Perform blend crossover on two parents to produce two children.

        The probability of crossover is determined by `crossover_rate`.
        A random alpha is used to blend the genes of the two parents.
        """
        child1 = parent1.copy()
        child2 = parent2.copy()
        if random.random() < crossover_rate:
            alpha = random.random()
            for i in range(num_parameters):
                child1[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
                child2[i] = alpha * parent2[i] + (1 - alpha) * parent1[i]
        return child1, child2

    def mutate(individual):
        """Mutate an individual's genes with some probability, using Gaussian noise."""
        for i in range(num_parameters):
            if random.random() < mutation_rate:
                noise = random.gauss(0.0, 0.05)  # standard deviation for mutation
                individual[i] += noise
                # Clamp to [init_min, init_max]
                individual[i] = max(min(individual[i], init_max), init_min)

    def select_parent(population, fitnesses):
        """Select a parent via tournament selection of size 3."""
        contenders = random.sample(range(len(population)), 3)
        best_index = contenders[0]
        for idx in contenders[1:]:
            if fitnesses[idx] > fitnesses[best_index]:
                best_index = idx
        return population[best_index]

    # Initialize population
    population = [create_individual() for _ in range(population_size)]
    fitnesses = [objective_function(ind) for ind in population]

    # Track best solution so far
    best_index = np.argmax(fitnesses)
    best_solution = population[best_index].copy()
    best_performance = fitnesses[best_index]

    # Main evolutionary loop
    for gen in range(max_generations):
        new_population = []
        new_fitnesses = []

        # Generate new individuals until we have a full population
        while len(new_population) < population_size:
            parent1 = select_parent(population, fitnesses)
            parent2 = select_parent(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            mutate(child1)
            mutate(child2)
            new_population.append(child1)
            new_population.append(child2)
            new_fitnesses.append(objective_function(child1))
            new_fitnesses.append(objective_function(child2))

        # Trim (if overshoot) and replace the old population
        population = new_population[:population_size]
        fitnesses = new_fitnesses[:population_size]

        # Update best known solution
        current_best = np.max(fitnesses)
        if current_best > best_performance:
            best_performance = current_best
            best_solution = population[np.argmax(fitnesses)].copy()

        print(f"[GA] Generation {gen + 1}/{max_generations}, "
              f"Best Performance: {best_performance:.4f}")

    return best_solution, best_performance
