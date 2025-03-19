# In-Situ Black Box Optimization Package

This package provides a modular framework for optimizing experimental setups using black‑box optimization techniques. It was developed with plasma metamaterial (PMM) optical computing experiments in mind, but the framework is general enough that you can plug in any objective function.

## Overview

The main script interactively asks you a series of questions about your experiment and objective function. Based on your answers (e.g., number of parameters, evaluation budget, noise level, smoothness, etc.), the script automatically suggest an optimizer. After suggesting an optimizer, the package asks if you would like to proceed with the suggestion or choose another method. 

You also have the option to use an example objective function or supply your own function via a Python module that defines a function called `user_objective`. This function should be in the Functions folder.

## Pip Installation

1. First make sure the main.py imports are written with <Optimizer.> in order to locate them in the folder.
2. Set up your virtual environment.
   ```bash
   python3 -m venv .venv
   ```
   ```bash
   source .venv/bin/activate
   ```
3. Install the package.
   ```bash
   pip install git+https://github.com/kbronstein56/Black-Box-Optimizer.git
   ```
5. Launch optimizer.
   ```bash
   python3 -m Optimizer.main
   ```

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/kbronstein56/Black-Box-Optimizer.git
   cd Black-Box-Optimizer
   ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Optimizers
- Genetic Algorithm: useful for high noise, descrete or large situations
- CMA-ES: useful for continuous, moderate dimension situations
- CMA-ES-GI: useful for smooth and high dimension situations
- Bayesian: useful for expensive functions and low dimensions
- Actor Critic: real-time adaptive control

## PMM-Specific Optimizations

For users working on plasma metamaterial (PMM) optical computing experiments, the `PMM_scripts` folder contains specialized scripts that implement the detailed hardware control, parameter mapping (e.g., using the Drude model), and experimental routines tailored to your setup.

*Note: Additional packages such as `minimalmodbus` or your hardware-specific modules are required for real experiments. For simulation, the package uses `FakePMMInSitu`.*

## Contact

For questions or further information, please contact:  
**Katherine P Bronstein** – bronstek@oregonstate.edu
