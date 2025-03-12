"""Actor-Critic Reinforcement Learning for In-Situ Optimization.

This module implements a simplified one-step Actor-Critic approach for black-box
optimization tasks. The actor is modeled as a Gaussian policy with a learned mean
vector and a fixed standard deviation. The critic is a simple linear function
that estimates the value of the current policy’s mean. The code returns the best
solution found and its performance.

Example:
    >>> import numpy as np
    >>> def user_objective(x):
    ...     # x is a NumPy array of parameters
    ...     return 40.0 - np.sum((x - 0.3)**2)
    >>> best_solution, best_perf = optimize_actor_critic(
    ...     objective_function=user_objective,
    ...     num_parameters=10,
    ...     episodes=50,
    ...     lr_actor=0.01,
    ...     lr_critic=0.01,
    ...     initial_std=0.1,
    ...     lower_bound=-1.0,
    ...     upper_bound=1.0
    ... )
    >>> print("Best solution:", best_solution)
    >>> print("Best performance:", best_perf)
"""

import numpy as np


def optimize_actor_critic(
    objective_function,
    num_parameters,
    episodes=100,
    lr_actor=0.01,
    lr_critic=0.01,
    initial_std=0.1,
    lower_bound=-1.0,
    upper_bound=1.0
):
    """Optimize a user-supplied objective function using a simple one-step Actor-Critic method.

    The actor:
      - Maintains a mean vector (actor_mean) of size `num_parameters`
      - Samples actions as actor_mean + normal_noise
      - Gradients are applied to update actor_mean based on advantage

    The critic:
      - Maintains a simple linear function v(x) = critic_weights^T * x + critic_bias
      - Updated via mean squared error to approximate the observed reward

    Args:
        objective_function (callable): A function f(x) -> float that takes a
            1D NumPy array (the parameters) and returns a scalar performance value
            (the higher, the better).
        num_parameters (int): Dimensionality of the parameter vector.
        episodes (int, optional): Number of one-step episodes to run.
        lr_actor (float, optional): Learning rate for updating the actor's mean.
        lr_critic (float, optional): Learning rate for updating the critic's weights.
        initial_std (float, optional): Fixed standard deviation for the policy’s
            Gaussian sampling.
        lower_bound (float, optional): Minimum allowed value for each parameter.
        upper_bound (float, optional): Maximum allowed value for each parameter.

    Returns:
        tuple:
            - np.ndarray: Best solution (parameter vector) found during training.
            - float: Best performance achieved by that solution.
    """
    # Actor parameters: mean vector and fixed std
    actor_mean = np.random.uniform(lower_bound, upper_bound, num_parameters)
    actor_std = initial_std

    # Critic parameters: linear function weights and bias
    critic_weights = np.random.randn(num_parameters) * 0.01
    critic_bias = 0.0

    # Evaluate initial solution
    best_solution = actor_mean.copy()
    best_performance = objective_function(actor_mean)

    def sample_action():
        """Sample an action (parameter vector) from the current actor policy."""
        noise = np.random.randn(num_parameters) * actor_std
        action = actor_mean + noise
        return np.clip(action, lower_bound, upper_bound)

    def critic_value(state):
        """Compute the critic's value estimate for a given state (actor's mean)."""
        return float(np.dot(critic_weights, state) + critic_bias)

    # Training loop
    for episode_index in range(episodes):
        # 1) Actor picks an action
        action = sample_action()

        # 2) Evaluate reward
        reward = objective_function(action)

        # 3) Critic estimates the value for current actor_mean
        value_estimate = critic_value(actor_mean)
        advantage = reward - value_estimate

        # 4) Update critic via gradient descent on MSE loss
        error = (value_estimate - reward)
        for i in range(num_parameters):
            grad_w = error * actor_mean[i]
            critic_weights[i] -= lr_critic * grad_w
        critic_bias -= lr_critic * error

        # 5) Update actor (policy gradient)
        grad_actor = (action - actor_mean) / (actor_std ** 2)
        actor_mean += lr_actor * advantage * grad_actor
        actor_mean = np.clip(actor_mean, lower_bound, upper_bound)

        # Track best solution
        if reward > best_performance:
            best_performance = reward
            best_solution = action.copy()

        print(
            f"[ActorCritic] Episode {episode_index + 1}/{episodes}: "
            f"Reward = {reward:.3f}, Advantage = {advantage:.3f}, "
            f"Best = {best_performance:.3f}"
        )

    return best_solution, best_performance
