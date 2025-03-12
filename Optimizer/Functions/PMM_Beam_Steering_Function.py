#!/usr/bin/env python3
"""
User Objective Function for Beam Steering Simulation

This module defines a function ``user_objective`` that evaluates the beam steering
performance for a plasma metamaterial device with 61 rods. It sets up the simulation
domain using InversePMMDesign’s PMMI class, runs a finite-difference frequency-domain
simulation (using the ceviche package), and computes an objective value based on the overlap
of the simulated fields with the probe measurements. The model is intended to be used as the
objective function in a black-box optimizer.

For a given rod parameter vector (rho), the objective function returns a performance
value (higher is better). In this example, an ideal configuration would yield an objective
value near 100.

Note: In this simulation, we assume that the ideal rod parameter value is 0.3. The functions
`Setup_Domain` and `Run_Evaluate` encapsulate the simulation setup and evaluation, respectively.
"""

import numpy as np
import math
import os

# Import required modules from InversePMMDesign and ceviche.
# For a real experiment, you might use a real hardware interface.
from PMM.PMMInverse import PMMI, mode_overlap, field_mag_int, c
from ceviche import fdfd_ez


def Setup_Domain():
    """Set up the simulation domain for beam steering evaluation.

    This function configures the simulation domain parameters, adds sources,
    probes, and designs the optimization region. It returns an instance of the
    PMMI class configured for the beam steering simulation.

    Returns:
        PMMI: The configured simulation domain object.
    """
    # Domain geometry parameters
    a = 0.020
    res = 16
    nx = 18
    ny = 22
    dpml = 2
    b_o = 0.0075 / a
    b_i = 0.0065 / a

    # Initialize the simulation domain.
    PPC = PMMI(a, res, nx, ny, dpml)

    # Add horn antennas (sources)
    PPC.Add_INFOMW_Horn(np.array([6.5, 11]), np.array([1, 0]), 6.5, pol='TM')
    PPC.Add_INFOMW_Horn(4 * np.array([0.5, np.sqrt(3) / 2]) + np.array([10.5, 11]),
                        np.array([-0.5, -np.sqrt(3) / 2]), 8, pol='TM')
    PPC.Add_INFOMW_Horn(4 * np.array([0.5, -np.sqrt(3) / 2]) + np.array([10.5, 11]),
                        np.array([-0.5, np.sqrt(3) / 2]), 8, pol='TM')

    # Specify the design region where rod parameters are optimized.
    PPC.Design_Region((6.5, 6.5), (8, 9))

    # Create a hexagonal rod array (plasma bulbs) in the design region.
    uniform = True
    PPC.Rod_Array_Hexagon_train(
        np.array([10.5, 11]), 5, b_i, 1,
        a_basis=np.array([[0, 1], [np.sqrt(3) / 2, 0.5]]),
        bulbs=True, r_bulb=(b_i, b_o), eps_bulb=3.8,
        uniform=uniform
    )

    # Set up sources and probes for the simulation.
    w = PPC.gamma(4.98e9)  # Source frequency (converted)
    wpmax = PPC.gamma(14e9)
    gamma = 0
    ew = 0.048 / a / 2 - 0.004 / a
    hd = 0.089 / a
    x = np.array([1, 0])
    y = np.array([0, 1])
    horn_dir_1 = np.array([0.5, np.sqrt(3) / 2])
    horn_dir_2 = np.array([0.5, -np.sqrt(3) / 2])
    open_dir_1 = np.array([np.sqrt(3) / 2, -0.5])
    open_dir_2 = np.array([np.sqrt(3) / 2, 0.5])
    cen = np.array([10.5, 11])

    PPC.Add_Source(np.array([6.5 - hd, 11 - ew]), np.array([6.5 - hd, 11 + ew]), w, 'src', 'ez')
    PPC.Add_Probe(
        (4 + hd) * horn_dir_1 + ew * open_dir_1 + cen,
        (4 + hd) * horn_dir_1 - ew * open_dir_1 + cen,
        w, 'prb1', 'ez'
    )
    PPC.Add_Probe(
        (4 + hd) * horn_dir_2 + ew * open_dir_2 + cen,
        (4 + hd) * horn_dir_2 - ew * open_dir_2 + cen,
        w, 'prb2', 'ez'
    )
    PPC.Add_Probe(
        -4.8 * (np.sqrt(3) / 2) * horn_dir_1 + 2.05 * open_dir_1 + cen,
        -4.8 * (np.sqrt(3) / 2) * horn_dir_1 - 2.05 * open_dir_1 + cen,
        w, 'loss_ul', 'ez'
    )
    PPC.Add_Probe(
        -4.8 * (np.sqrt(3) / 2) * horn_dir_2 + 2.05 * open_dir_2 + cen,
        -4.8 * (np.sqrt(3) / 2) * horn_dir_2 - 2.05 * open_dir_2 + cen,
        w, 'loss_ll', 'ez'
    )
    PPC.Add_Probe(
        4.8 * (np.sqrt(3) / 2) * x - 2.05 * y + cen,
        4.8 * (np.sqrt(3) / 2) * x + 2.05 * y + cen,
        w, 'loss_R', 'ez'
    )
    return PPC


def Run_Evaluate(PPC, rho, E0=None, E0l=None):
    """Run the beam steering simulation for a given rod configuration.

    This function converts the rod parameters (rho) into the corresponding permittivity
    distribution using the PMMI object's parameterization method, runs the finite-difference
    simulation via the ceviche solver, and computes an objective value based on mode overlap
    and field magnitude integrals from various probes.

    Args:
        PPC: The simulation domain object returned by Setup_Domain().
        rho (np.ndarray): A 1D array of rod parameters (expected length 61).
        E0: Baseline overlap from the primary probe (optional).
        E0l: List of baseline field magnitudes from other probes (optional).

    Returns:
        tuple: A tuple (objective, E0, E0l) where:
            - objective (float): The computed performance value.
            - E0: The baseline overlap value.
            - E0l: A list of baseline field magnitudes.
    """
    if E0l is None:
        E0l = []
    epsr_init = PPC.Rho_Parameterization_wp(
        rho,
        PPC.sources['src'][1] * PPC.a / (2 * np.pi * c),
        wp_max=PPC.gamma(14e9), gamma=0, uniform=True
    )
    sim = fdfd_ez(PPC.sources['src'][1], PPC.dl, epsr_init, [PPC.Npml, PPC.Npml])
    _, _, E = sim.solve(PPC.sources['src'][0])

    if E0 is None:
        E0 = mode_overlap(E, PPC.probes['prb1'][0])
    if not E0l:
        for prb_key in ['prb2', 'loss_ul', 'loss_ll', 'loss_R']:
            E0l.append(field_mag_int(E, PPC.probes[prb_key][3]))

    objective = mode_overlap(E, PPC.probes['prb1'][0]) / E0
    i = 0
    for prb_key in ['prb2', 'loss_ul', 'loss_ll', 'loss_R']:
        objective -= 2 * field_mag_int(E, PPC.probes[prb_key][3]) / E0l[i]
        i += 1
    return objective, E0, E0l


def user_objective(rod_parameters: np.ndarray) -> float:
    """Evaluate the beam steering performance for a given rod configuration.

    This function sets up the simulation domain, runs the beam steering simulation using the
    provided rod parameters, and returns a performance value. The expected input is a NumPy
    array of 61 parameters. An optimal configuration (for example, all parameters near 0.3)
    should yield a high performance value (ideally near 100 in this simulation).

    Args:
        rod_parameters (np.ndarray): A 1D array of rod parameters (length 61).

    Returns:
        float: The computed performance value from the simulation (higher is better).
    """
    # Ensure the input vector has the correct length.
    if len(rod_parameters) != 61:
        raise ValueError("Expected 61 rod parameters, got {}.".format(len(rod_parameters)))
    
    # Set up the simulation domain.
    PPC = Setup_Domain()
    
    # Run the simulation evaluation.
    objective_value, _, _ = Run_Evaluate(PPC, rod_parameters)
    return objective_value


if __name__ == "__main__":
    # For testing: evaluate the objective on an initial configuration where all rods are set to 0.3.
    initial_configuration = np.full(61, 0.3)
    performance_value = user_objective(initial_configuration)
    print("Performance for initial configuration: {:.4f}".format(performance_value))
