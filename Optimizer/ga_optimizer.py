"""
Genetic Algorithm Optimizer

This module provides a function to optimize a user-supplied objective function
using a Genetic Algorithm (GA) with advanced operators including stall detection
and extreme mutation. Each individual is a vector of parameters, and the GA employs:
  - Tournament selection
  - Blend crossover
  - Gaussian mutation
  - Stall detection with extreme mutation to reintroduce diversity

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
    mutation_rate=0.1,
    stall_limit=2,
    extreme_mutation_sigma=0.2,
    extreme_mutation_fraction=1.0
):
    """Optimize a given objective function using a Genetic Algorithm (GA) with advanced operators.

    This function initializes a population of individuals (parameter vectors), evolves them
    over several generations, and returns the best solution along with its performance. In addition
    to standard operators (tournament selection, blend crossover, and Gaussian mutation), the algorithm
    monitors progress; if no improvement is seen for a certain number of generations (stall_limit),
    it applies an extreme mutation to all individuals to help escape local minima.

    Args:
        objective_function (callable): A function f(x) -> float that evaluates the performance of a
            parameter vector (higher is better).
        num_parameters (int): Dimensionality of the parameter space.
        population_size (int, optional): Number of individuals in the GA population.
        max_generations (int, optional): Number of generations to run the GA.
        init_min (float, optional): Lower bound for initializing parameters.
        init_max (float, optional): Upper bound for initializing parameters.
        crossover_rate (float, optional): Probability that crossover is applied.
        mutation_rate (float, optional): Probability of mutating each gene.
        stall_limit (int, optional): Number of generations with no improvement before triggering extreme mutation.
        extreme_mutation_sigma (float, optional): Standard deviation used for extreme mutation.
        extreme_mutation_fraction (float, optional): Fraction of genes to be mutated during extreme mutation.

    Returns:
        tuple:
            - np.ndarray: The best solution (parameter vector) found.
            - float: The performance (fitness) of that best solution.
    """

    def create_individual():
        """Create a random individual within the [init_min, init_max] range."""
        return np.array([random.uniform(init_min, init_max) for _ in range(num_parameters)])

    def crossover(parent1, parent2):
        """Perform blend crossover on two parents to produce two children."""
        child1 = parent1.copy()
        child2 = parent2.copy()
        if random.random() < crossover_rate:
            alpha = random.random()
            for i in range(num_parameters):
                child1[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
                child2[i] = alpha * parent2[i] + (1 - alpha) * parent1[i]
        return child1, child2

    def mutate(individual):
        """Mutate each gene with probability mutation_rate using Gaussian noise, then clamp."""
        for i in range(num_parameters):
            if random.random() < mutation_rate:
                noise = random.gauss(0.0, 0.05)
                individual[i] += noise
                individual[i] = max(min(individual[i], init_max), init_min)

    def mutate_extreme(individual, sigma, fraction):
        """Apply extreme mutation to a given individual on a fraction of genes."""
        num_genes = len(individual)
        num_to_mutate = int(num_genes * fraction)
        indices = random.sample(range(num_genes), num_to_mutate)
        for i in indices:
            noise = random.gauss(0.0, sigma)
            individual[i] += noise
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

    best_index = np.argmax(fitnesses)
    best_solution = population[best_index].copy()
    best_performance = fitnesses[best_index]
    stall_counter = 0

    # Main evolutionary loop
    for gen in range(max_generations):
        new_population = []
        new_fitnesses = []
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

        population = new_population[:population_size]
        fitnesses = new_fitnesses[:population_size]

        current_best = np.max(fitnesses)
        if current_best > best_performance:
            best_performance = current_best
            best_solution = population[np.argmax(fitnesses)].copy()
            stall_counter = 0  # reset stall counter on improvement
        else:
            stall_counter += 1

        # If stalled for too long, apply extreme mutation to all individuals
        if stall_counter >= stall_limit:
            print(f"[GA] Stall detected at generation {gen+1}. Applying extreme mutation.")
            for ind in population:
                mutate_extreme(ind, extreme_mutation_sigma, extreme_mutation_fraction)
            # Re-evaluate population after extreme mutation
            fitnesses = [objective_function(ind) for ind in population]
            if np.max(fitnesses) > best_performance:
                best_performance = np.max(fitnesses)
                best_solution = population[np.argmax(fitnesses)].copy()
            stall_counter = 0

        print(f"[GA] Generation {gen+1}/{max_generations}, Best Performance: {best_performance:.4f}")

    return best_solution, best_performance
