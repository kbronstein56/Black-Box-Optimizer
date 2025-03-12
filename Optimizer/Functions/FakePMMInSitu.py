"""A simplified simulation of a plasma metamaterial (PMM) environment for testing.

This module provides a mock environment that mimics the behavior of a real
plasma metamaterial (PMM) device. It is meant for offline testing of
optimization algorithms and does not accurately represent real physics.

Example:
    >>> from fake_pmm_in_situ import FakePMMInSitu
    >>> pm = FakePMMInSitu()
    >>> pm.Config_Warmup(T=2)
    >>> test_rho = np.full(100, 0.3)
    >>> pm.ArraySet_Rho(test_rho, 7.0, knob=0.5, scale=1.0)
    >>> performance, _ = pm.Wvg_Obj_Get(test_rho, fpm=7.0, k=0.5, S=1.0,
    ...                                 f=5.0, df=0.25, objective='dB',
    ...                                 norms=[], duty_cycle=0.5)
    >>> print("Performance:", performance)

Notes:
    - The default ideal solution in this mock environment has each rod set to 0.3.
    - The performance function is 40 - sum((rho - 0.3)^2), plus a small Gaussian noise.
    - The class methods mimic the interface of a real PMMInSitu device but do not
      accurately represent actual hardware behavior.

Attributes:
    num_rods (int): The number of rods in the device (default 100).
    ideal_rho (float): The ideal rod value that yields maximum performance (default 0.3).
    stored_rho (np.ndarray): Last set of rod parameters by ArraySet_Rho.
"""

import numpy as np
import time


class FakePMMInSitu:
    """A mock PMMInSitu-like class for offline testing of optimizers.

    This class simulates a plasma metamaterial device with a fixed ideal rod
    configuration at 0.3. The performance metric is artificially defined as
    40 minus the sum of squared errors from the ideal value, plus small noise.

    Args:
        conf_file (str, optional): Configuration file path for an actual PMM
            device. Ignored in this mock class.
    """

    def __init__(self, conf_file=None):
        """Initialize the FakePMMInSitu mock environment.

        Args:
            conf_file (str, optional): Configuration file path. Defaults to None.
                In this mock class, the argument is ignored.
        """
        self.num_rods = 100
        print("[FakePMMInSitu] Initialized in simulation mode.")

        self.ideal_rho = 0.3
        self.stored_rho = None

    def Config_Warmup(self, T=2, ballasts='New', duty_cycle=0.5):
        """Pretend to warm up the device.

        In a real scenario, this might involve turning on power supplies or
        stabilizing conditions. Here, we only emulate a short sleep.

        Args:
            T (int, optional): Number of minutes to emulate warmup. Defaults to 2.
            ballasts (str, optional): Ballast type or configuration. Ignored here.
            duty_cycle (float, optional): Fraction of active time. Ignored in this mock.
        """
        print("[FakePMMInSitu] Pretending to warm up for", T, "minutes.")
        time.sleep(0.01)

    def ArraySet_Rho(self, rho, wp_max, knob=0.5, scale=1.0):
        """Pretend to set rod currents based on the parameter array.

        In an actual device, this would involve setting power supplies or
        controlling discharge currents. Here, we only store the parameter array.

        Args:
            rho (np.ndarray): Array of length `num_rods` specifying rod settings.
            wp_max (float): Approximate maximum nondimensionalized plasma frequency.
                Ignored in this mock.
            knob (float, optional): A knob parameter for scaling. Ignored here.
            scale (float, optional): Additional scaling factor. Ignored here.
        """
        self.stored_rho = rho.copy()
        time.sleep(0.05)  # Emulate a small delay for hardware interaction

    def Deactivate_Bulb(self, who='all'):
        """Pretend to deactivate power supplies for rods.

        Args:
            who (str, optional): Identifier for which bulbs to deactivate. Defaults to 'all'.
        """
        return

    def f_a(self, freq_ghz):
        """Return the nondimensional frequency for a given GHz input.

        In a real device, this would convert a frequency in GHz to a unitless
        parameter (omega * a / c). Here, we simply return `freq_ghz`.

        Args:
            freq_ghz (float): Frequency in GHz.

        Returns:
            float: The same input value (`freq_ghz`) in this mock.
        """
        return freq_ghz

    def Get_S21_S31(self):
        """Simulate retrieval of S-parameters in the 'fake' environment.

        In a real PMMInSitu class, this might query a vector network analyzer to get
        S21 and S31 at different frequencies. Here, we return dummy values.

        Returns:
            tuple:
                - freq (np.ndarray): A dummy array with a single frequency value [5.0].
                - S21 (np.ndarray): A dummy array [0.0].
                - S31 (np.ndarray): A dummy array [0.0].
        """
        freq = np.array([5.0])
        S21 = np.array([0.0])
        S31 = np.array([0.0])
        return freq, S21, S31

    def Wvg_Obj_Get(self, rho, fpm, k, S, f, df, objective='dB', norms=None, duty_cycle=0.5):
        """Compute a fake waveguide objective based on rod configuration.

        The performance function is defined as:
          40.0 - sum((rho - 0.3)^2) + Gaussian noise(0, 0.2).

        Args:
            rho (np.ndarray): Rod parameters for which performance is evaluated.
            fpm (float): Plasma frequency max in GHz. Ignored in the mock.
            k (float): A knob parameter. Ignored in the mock.
            S (float): Additional scale factor. Ignored here.
            f (float): Frequency of interest in GHz. Ignored in the mock.
            df (float): Frequency bandwidth. Ignored in the mock.
            objective (str, optional): Type of measurement. Defaults to 'dB'. Ignored here.
            norms (list, optional): Placeholder for normalization factors. Defaults to None.
            duty_cycle (float, optional): Not used in this mock environment.

        Returns:
            tuple:
                - performance (float): The computed performance score.
                - empty_list (list): An empty list, placeholder for additional data.
        """
        if norms is None:
            norms = []

        diff = rho - self.ideal_rho
        base_score = 40.0 - np.sum(diff ** 2)
        noise = np.random.normal(0, 0.2)
        performance = base_score + noise

        return performance, []
