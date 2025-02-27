# In-Situ Black Box Optimization Package

This package provides a modular framework for optimizing experimental setups using black‑box optimization techniques. It was developed with plasma metamaterial (PMM) optical computing experiments in mind, but the framework is general enough that you can plug in any objective function.

## Overview

The main script interactively asks you a series of questions about your experiment and objective function. Based on your answers (e.g., number of parameters, evaluation budget, noise level, smoothness, etc.), the script automatically suggest an optimizer.

After suggesting an optimizer, the package asks if you would like to proceed with the suggestion or choose another method. You also have the option to use an example objective function or supply your own function (via a Python module that defines a function called `user_objective`).

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://your.repository.url/in_situ_optimization.git
   cd in_situ_optimization
   ```

2. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```


## PMM-Specific Optimizations

For users working on plasma metamaterial (PMM) optical computing experiments, the `pmm_optimizations` folder contains specialized scripts that implement the detailed hardware control, parameter mapping (e.g., using the Drude model), and experimental routines tailored to your setup.

*Note: Additional packages such as `minimalmodbus` or your hardware-specific modules are required for real experiments. For simulation, the package uses `FakePMMInSitu`.*

## Contact

For questions or further information, please contact:  
**Katherine P Bronstein** – bronstek@oregonstate.edu
