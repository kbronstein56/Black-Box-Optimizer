"""
User Objective Function for Plasma Metamaterial Devices

This module defines a function `user_objective` which interfaces with the PMM experiment.
It sets the rod configuration (rho) using ArraySet_Rho and measures the waveguide performance
using Wvg_Obj_Get. This is meant for in-situ optimization of plasma metamaterial devices.
For real hardware, uncomment the PMMInSitu import and related lines.

Note:
    - Currently, it uses FakePMMInSitu for simulation.
    - For real experiments, replace FakePMMInSitu with PMMInSitu and adjust file paths/configurations.
"""

import numpy as np
import time
# For real hardware, uncomment the following line:
# from PMMInSitu import PMMInSitu
from FakePMMInSitu import FakePMMInSitu  # for simulation

def user_objective(parameters: np.ndarray) -> float:
    """Evaluate the performance of a given rod configuration for a plasma metamaterial device.

    This function sets the rod parameters using ArraySet_Rho, measures the waveguide performance
    using Wvg_Obj_Get, and then deactivates the bulbs. It returns the measured performance.

    Args:
        parameters (np.ndarray): A 1D NumPy array representing the rod parameters (rho values).

    Returns:
        float: The measured performance (higher is better).
    """
    # Use the fake PMM environment for simulation.
    # For real hardware, replace FakePMMInSitu with PMMInSitu and provide a valid config file.
    pmm = FakePMMInSitu()
    pmm.Config_Warmup(T=2)  # For simulation/demo; adjust or remove for real-time operation

    # Set rod parameters in the PMM device.
    pmm.ArraySet_Rho(parameters, pmm.f_a(7.0), knob=0.5, scale=1.0)
    # If using real hardware, you might remove or adjust the delay below.
    time.sleep(0.01)

    # Measure waveguide performance.
    performance, _ = pmm.Wvg_Obj_Get(
        parameters,
        fpm=7.0,
        k=0.5,
        S=1.0,
        f=5.0,
        df=0.25,
        objective='dB',
        norms=[],
        duty_cycle=0.5
    )
    # Deactivate bulbs after measurement.
    pmm.Deactivate_Bulb('all')
    # Return the performance; higher means a better configuration.
    return performance

if __name__ == "__main__":
    # For testing purposes, run the objective on a sample parameter vector.
    sample_parameters = np.array([0.3] * 100)
    perf = user_objective(sample_parameters)
    print(f"Sample performance: {perf:.4f}")
