"""
Bayesian Optimization for In-Situ PMM Waveguide Tuning
Using the bayesian-optimization library: pip install bayesian-optimization

We have 100 rods, so we define 100 parameters "rod0" to "rod99" each in [-1,1].
We feed them to FakePMMInSitu or real PMMInSitu and measure performance.

Then define waveguide_performance(...). Then do a BayesianOptimization
with 'init_points' random initial tries, then 'n_iter' sequential tries.
"""
import time
import numpy as np
import random
from bayes_opt import BayesianOptimization

# ------------------------------------------------------------------
# Switch between real device or fake data
# ------------------------------------------------------------------
# from PMMInSitu import PMMInSitu   # real device
from FakePMMInSitu import FakePMMInSitu  # for a simulated environment

# ------------------------------------------------------------------
# Global Settings
# ------------------------------------------------------------------
NUM_RODS = 100
INIT_MIN = -1.0
INIT_MAX =  1.0
INIT_POINTS = 5   # how many random initial points
N_ITER = 25       # how many Bayesian Optimization steps after that
                  # (note: large dimension => these might not go far, but for demo)

# waveguide objective knobs
FPM   = 7.0  
KNOB  = 0.5  
SCALE = 1.0  
F_TARG= 5.0  
DF_BW = 0.25 
DUTY  = 0.5  

# ------------------------------------------------------------------
# waveguide performance function
#  "maximize waveguide performance" => we directly return performance
# ------------------------------------------------------------------
def waveguide_performance(pm: FakePMMInSitu, rod_params: np.ndarray) -> float:
    """
    We set the rods to rod_params, measure waveguide performance,
    and return the performance as a float (the bigger, the better).
    """
    # 1) physically set rods in [rod_params].
    pm.ArraySet_Rho(rod_params, pm.f_a(FPM), knob=KNOB, scale=SCALE)

    # 2) measure waveguide performance
    performance, _ = pm.Wvg_Obj_Get(
        rod_params,
        fpm=FPM,
        k=KNOB,
        S=SCALE,
        f=F_TARG,
        df=DF_BW,
        objective='dB',
        norms=[],
        duty_cycle=DUTY
    )
    # Deactivate for housekeeping
    pm.Deactivate_Bulb('all')

    return performance

# ------------------------------------------------------------------
# the "black-box" function for bayesian-optimization
# we'll define param rod0..rod99
# ------------------------------------------------------------------
def waveguide_BO_function(**kwargs):
    """
    This function is called by the BayesianOptimization library.
    'kwargs' is a dict { 'rod0': value, 'rod1': value, ..., 'rod99':value }
    Each in [INIT_MIN, INIT_MAX].
    We'll convert to a param array and measure waveguide performance.
    Return the performance (the library tries to maximize it).
    """
    # convert kwargs -> param array
    param_array = np.zeros(NUM_RODS)
    for i in range(NUM_RODS):
        key = f"rod{i}"
        param_array[i] = kwargs[key]

    # measure performance using global pm object
    global pm
    performance_value = waveguide_performance(pm, param_array)
    return performance_value

# ------------------------------------------------------------------
# main script
# ------------------------------------------------------------------
def main():
    # record time
    start_time = time.time()

    # create the device object
    # if real hardware: from PMMInSitu import PMMInSitu
    # pm = PMMInSitu("./../confs/conf_test.yaml")
    # or use the fake data:
    global pm  # so waveguide_BO_function can see it
    pm = FakePMMInSitu()
    print("Initialized PMMInSitu or FakePMMInSitu. Warmup demonstration:")
    pm.Config_Warmup(T=2)

    # define the param bounds for Bayesian Optimization:
    # "rod0": (INIT_MIN, INIT_MAX), "rod1":..., etc
    rod_bounds = {}
    for i in range(NUM_RODS):
        rod_bounds[f"rod{i}"] = (INIT_MIN, INIT_MAX)

    # create BayesianOptimization object
    optimizer = BayesianOptimization(
        f=waveguide_BO_function,  # the black-box function
        pbounds=rod_bounds,       # param bounds
        verbose=2,               # verbose=2 prints more logs
        random_state=None        # or a fixed int for reproducibility
    )

    # run .maximize with some init_points + n_iter
    # init_points => random exploration
    # n_iter => subsequent tries with surrogate model
    optimizer.maximize(
        init_points=INIT_POINTS,
        n_iter=N_ITER
    )

    # after finishing:
    best_solution = optimizer.max['params']   # dictionary of rod0..rod99
    best_performance = optimizer.max['target'] # best performance found

    print("\n[BayesianOptimization] Finished.")
    # convert best_solution dict -> array
    best_rho = np.array([ best_solution[f"rod{i}"] for i in range(NUM_RODS) ])
    print(f"Best performance found = {best_performance:.4f}")
    print("Best rods array:")
    print(best_rho)

    # set device to best solution
    pm.ArraySet_Rho(best_rho, pm.f_a(FPM), knob=KNOB, scale=SCALE)
    time.sleep(0.05)
    print("[BayesianOptimization] Device set to best solution.")

    # measure total time
    end_time = time.time()
    elapsed = end_time - start_time
    hh = int(elapsed // 3600)
    mm = int((elapsed % 3600) // 60)
    ss = int(elapsed % 60)
    print(f"[MAIN] Total run time: {hh}h:{mm}m:{ss}s")


if __name__ == "__main__":
    main()
