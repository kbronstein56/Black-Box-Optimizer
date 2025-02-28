"""
User Objective Function for Viscosity Curve Fitting

This module defines a function `user_objective` which computes the performance of a two-parameter 
model for fitting viscosity data. The viscosity data are read from a CSV file ("user_curve_fitting_viscosity_data.csv")
that contains shear rate and measured viscosity values. The model assumes the following form:

    viscosity_predicted = a * exp(-b * shear_rate)

where the parameter vector is x = [a, b]. The objective function computes the sum of squared 
errors between the predicted viscosity and the measured viscosity, and returns the negative error 
(as performance, so that a lower error corresponds to a higher performance).

The ideal (optimal) performance is achieved when the predicted values closely match the data.
For example, if the data were generated using a=50 and b=0.05 (with little noise), the optimizer 
should converge to parameter values near these and performance near zero.

Expected user inputs:
    Enter the number fo parameters in your experiment:n2
    How many function ealuations can you experiment support: 1000
    What is the noise level: moderate
    Is your objective function smooth and continuous: yes
    Custom Bounds: lower= 40,0 and upper= 60, 0.1
    --> Use Bayesian or CMA-ES
"""

import numpy as np
import pandas as pd
import math

def user_objective(parameters: np.ndarray) -> float:
    """
    Compute the performance of the viscosity curve fitting model.

    This function reads viscosity data from 'user_curve_fitting_viscosity_data.csv', applies a two-parameter model,
    and computes the negative sum of squared errors between the predicted viscosity and the measured
    viscosity. A higher (less negative) performance indicates a better fit.

    The model is:
        viscosity_predicted = a * exp(-b * shear_rate)
    where parameters[0] = a and parameters[1] = b.

    Returns:
        float: The performance value (negative sum of squared errors).
    """
    try:
        data = pd.read_csv("user_curve_fitting_viscosity_data.csv")
    except Exception as e:
        raise RuntimeError("Error reading 'user_curve_fitting_viscosity_data.csv': " + str(e))

    # Extract data as NumPy arrays.
    shear_rate = data["shear_rate"].values
    measured_viscosity = data["viscosity"].values

    a_parameter = parameters[0]
    b_parameter = parameters[1]

    predicted_viscosity = a_parameter * np.exp(-b_parameter * shear_rate)

    squared_errors = (predicted_viscosity - measured_viscosity) ** 2
    error_sum = np.sum(squared_errors)

    performance = -error_sum
    return performance

if __name__ == "__main__":
    # For testing purposes, run the objective on a sample parameter vector.
    sample_parameters = np.array([50.0, 0.05])
    performance_value = user_objective(sample_parameters)
    print(f"Sample performance: {performance_value:.4f}")
