"""
Example user objective function.

Define a function named 'user_objective' that accepts a NumPy array of parameters
and returns a scalar performance value (where higher is better).

Example:
    Optimal performance of 40 is achieved when every parameter equals 0.3.
"""

import numpy as np

def user_objective(x):
    ideal_value = 0.3
    error = np.sum((x - ideal_value) ** 2)
    performance = 40 - error
    return performance
