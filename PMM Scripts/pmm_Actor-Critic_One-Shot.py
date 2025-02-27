"""
Simplified  one-step environment Actor-Critic approach for in-situ waveguide 
optimization with 100 rods

This script optimizes the configuration (rod currents) of a 100-rod plasma
metamaterial device using an Actor–Critic approach in a reduced latent space.
The actor learns a latent mean and log standard deviation (for a Gaussian in
a 10-dimensional subspace) and also learns a linear mapping (decoder) from that
latent space to the 100-dimensional rod space. The critic is a simple linear
function on the latent mean.  We run multiple episodes. Each "episode" is one 
attempt at setting rods,measuring performance as reward, then updating both 
the actor and critic via a gradient-based method.

  - A "policy" (actor) that outputs 100 rod parameters (the action)
  - A "value function" (critic) that estimates the expected performance
  - Run episodes


Low dimensional subspace for the actor’s mean vector might help 
for high dimensions by reducing the parameter space that the actor tries to learn.
Latent dimension is 10, so we have a 10D “policy” and a 100×10 decoder.

RL approach is too simple for the "FakeData." It might be useful for experiment
because we can run thousands of trials.
It may become very useful in the future for a temporally responsive device.
"""

import numpy as np
import math
import time
import random

# Uncomment one of these based on your setup:
# from PMMInSitu import PMMInSitu      # For real hardware
from FakePMMInSitu import FakePMMInSitu  # For simulated environment

# ==================== Global Settings ====================
NUMBER_OF_RODS = 100
LATENT_DIMENSION = 10

LOWER_BOUND_FOR_RODS = -1.0
UPPER_BOUND_FOR_RODS = 1.0

NUMBER_OF_EPISODES = 100

LEARNING_RATE_ACTOR = 0.01
LEARNING_RATE_CRITIC = 0.01
LEARNING_RATE_DECODER = 0.0005

INITIAL_STD_IN_LATENT = 0.05
CLAMP_LOG_STD_LOW = -10.0
CLAMP_LOG_STD_HIGH = 3.0

# Waveguide objective knobs
MAX_PLASMA_FREQUENCY = 7.0
KNOB_PARAMETER = 0.5
SCALE_PARAMETER = 1.0
TARGET_FREQUENCY = 5.0
BANDWIDTH = 0.25
DUTY_CYCLE = 0.5

# ==================== Utility Functions ====================
def clamp_value(value, min_val, max_val):
    """Return the value clamped between min_val and max_val."""
    if value < min_val:
        return min_val
    elif value > max_val:
        return max_val
    else:
        return value

def clamp_array(arr, min_val, max_val):
    """Clamp each element of a numpy array to [min_val, max_val]."""
    clamped = np.copy(arr)
    for i in range(len(arr)):
        clamped[i] = clamp_value(arr[i], min_val, max_val)
    return clamped

# ==================== Waveguide Performance Function ====================
def measure_waveguide_performance(pmm_env, rod_parameters):
    """
    Set the rod parameters in the PMM environment and return the measured performance.
    Higher performance values are better.
    """
    frequency_converted = pmm_env.f_a(MAX_PLASMA_FREQUENCY)
    pmm_env.ArraySet_Rho(rod_parameters, frequency_converted,
                         knob=KNOB_PARAMETER, scale=SCALE_PARAMETER)
    # No delay added
    performance_value, _ = pmm_env.Wvg_Obj_Get(
        rod_parameters,
        fpm=MAX_PLASMA_FREQUENCY,
        k=KNOB_PARAMETER,
        S=SCALE_PARAMETER,
        f=TARGET_FREQUENCY,
        df=BANDWIDTH,
        objective='dB',
        norms=[],
        duty_cycle=DUTY_CYCLE
    )
    pmm_env.Deactivate_Bulb('all')
    return performance_value

# ==================== Actor with Learned Decoder ====================
class LatentActorWithDecoder:
    """
    The actor uses a low-dimensional latent space (of size LATENT_DIMENSION) to generate
    a 100-dimensional rod configuration. It maintains:
      - latent_mean_vector (R^LATENT_DIMENSION)
      - latent_log_std (R^LATENT_DIMENSION)
      - rod_decoder_matrix (shape [NUMBER_OF_RODS, LATENT_DIMENSION])
      - rod_decoder_bias (shape [NUMBER_OF_RODS])
    
    When sampling an action:
      z ~ N(latent_mean_vector, diag(exp(2 * latent_log_std)))
      rods = clamp(rod_decoder_matrix @ z + rod_decoder_bias, LOWER_BOUND_FOR_RODS, UPPER_BOUND_FOR_RODS)
    """
    def __init__(self, dimension_of_rods, latent_dimension, initial_log_std):
        self.dimension_of_rods = dimension_of_rods
        self.latent_dimension = latent_dimension
        self.latent_mean_vector = np.zeros(self.latent_dimension, dtype=float)
        self.latent_log_std = np.full(self.latent_dimension, math.log(initial_log_std), dtype=float)
        self.rod_decoder_matrix = np.random.randn(self.dimension_of_rods, self.latent_dimension) * 0.01
        self.rod_decoder_bias = np.zeros(self.dimension_of_rods, dtype=float)

    def decode_latent_to_rods(self, latent_vector):
        raw_rods = np.dot(self.rod_decoder_matrix, latent_vector) + self.rod_decoder_bias
        rods = np.clip(raw_rods, LOWER_BOUND_FOR_RODS, UPPER_BOUND_FOR_RODS)
        return rods

    def sample_action_with_latent(self):
        z_sample = np.zeros(self.latent_dimension, dtype=float)
        for i in range(self.latent_dimension):
            std_i = math.exp(self.latent_log_std[i])
            z_sample[i] = self.latent_mean_vector[i] + std_i * np.random.randn()
        rod_action = self.decode_latent_to_rods(z_sample)
        return rod_action, z_sample

    def log_prob_of_latent(self, z_sample):
        diff = z_sample - self.latent_mean_vector
        total_log_prob = 0.0
        for i in range(self.latent_dimension):
            std_i = math.exp(self.latent_log_std[i])
            log_prob_i = -0.5 * ((diff[i] / std_i) ** 2 + math.log(2 * math.pi * (std_i ** 2)))
            total_log_prob += log_prob_i
        return total_log_prob

    def update_actor(self, z_sample, rod_action, advantage, learning_rate_actor, learning_rate_decoder):
        # Update latent parameters (mean and log_std)
        for i in range(self.latent_dimension):
            std_i = math.exp(self.latent_log_std[i])
            grad_mean = (z_sample[i] - self.latent_mean_vector[i]) / (std_i ** 2)
            self.latent_mean_vector[i] += learning_rate_actor * advantage * grad_mean

            grad_log_std = ((z_sample[i] - self.latent_mean_vector[i]) ** 2) / (std_i ** 2) - 1.0
            self.latent_log_std[i] += learning_rate_actor * advantage * 0.1 * grad_log_std

        self.latent_log_std = np.clip(self.latent_log_std, CLAMP_LOG_STD_LOW, CLAMP_LOG_STD_HIGH)

        # Update the decoder parameters using a regression-like update:
        predicted_rods = self.decode_latent_to_rods(z_sample)
        decoder_error = rod_action - predicted_rods  # shape: [NUMBER_OF_RODS]
        for i in range(self.dimension_of_rods):
            for j in range(self.latent_dimension):
                self.rod_decoder_matrix[i, j] += learning_rate_decoder * advantage * decoder_error[i] * z_sample[j]
            self.rod_decoder_bias[i] += learning_rate_decoder * advantage * decoder_error[i]

# ==================== Critic ====================
class SimpleCritic:
    """
    The critic estimates the value function as a linear function of the actor's latent mean:
      V = dot(weights, latent_mean_vector) + bias.
    """
    def __init__(self, latent_dimension):
        self.latent_dimension = latent_dimension
        self.weights = np.random.randn(latent_dimension) * 0.01
        self.bias = 0.0

    def value_estimate(self, latent_mean_vector):
        return float(np.dot(self.weights, latent_mean_vector) + self.bias)

    def update_critic(self, latent_mean_vector, observed_return, learning_rate_critic):
        v_estimate = self.value_estimate(latent_mean_vector)
        error = v_estimate - observed_return
        for i in range(self.latent_dimension):
            grad_w = error * latent_mean_vector[i]
            self.weights[i] -= learning_rate_critic * grad_w
        self.bias -= learning_rate_critic * error

# ==================== Main Loop ====================
def main():
    pmm_env = FakePMMInSitu()  # For simulation; replace with PMMInSitu if using real hardware.
    pmm_env.Config_Warmup(T=2)

    actor = LatentActorWithDecoder(
        dimension_of_rods=NUMBER_OF_RODS,
        latent_dimension=LATENT_DIMENSION,
        initial_log_std=INITIAL_STD_IN_LATENT
    )
    critic = SimpleCritic(latent_dimension=LATENT_DIMENSION)

    current_lr_actor = LEARNING_RATE_ACTOR
    current_lr_critic = LEARNING_RATE_CRITIC
    current_lr_decoder = LEARNING_RATE_DECODER

    start_time = time.time()
    best_performance = -1e9
    best_rod_configuration = None

    for episode in range(NUMBER_OF_EPISODES):
        rod_configuration, latent_z = actor.sample_action_with_latent()
        performance_reward = measure_waveguide_performance(pmm_env, rod_configuration)
        critic_estimate = critic.value_estimate(actor.latent_mean_vector)
        advantage = performance_reward - critic_estimate

        critic.update_critic(actor.latent_mean_vector, performance_reward, current_lr_critic)
        actor.update_actor(latent_z, rod_configuration, advantage, current_lr_actor, current_lr_decoder)

        current_lr_actor *= 0.995
        current_lr_critic *= 0.995
        current_lr_decoder *= 0.995

        if performance_reward > best_performance:
            best_performance = performance_reward
            best_rod_configuration = rod_configuration.copy()

        print(f"Episode {episode+1}/{NUMBER_OF_EPISODES}: Reward = {performance_reward:.3f}, "
              f"Advantage = {advantage:.3f}, Best so far = {best_performance:.3f}")

    total_runtime = time.time() - start_time

    print("\n[Actor-Critic Full Subspace with Decoder Learning] Finished.")
    print(f"Best performance found = {best_performance:.4f}")
    print("Best rods configuration:")
    print(best_rod_configuration)

    pmm_env.ArraySet_Rho(best_rod_configuration, pmm_env.f_a(MAX_PLASMA_FREQUENCY),
                         knob=KNOB_PARAMETER, scale=SCALE_PARAMETER)
    print("[Actor-Critic Full] Device set to best solution.")
    print(f"Total run time: {total_runtime:.2f} s")

if __name__ == "__main__":
    main()
