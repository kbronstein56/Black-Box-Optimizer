# ------------------------------------------------------------------
# CMA-ES for In-Situ PMM Optimization
# ------------------------------------------------------------------
"""
The idea is to optimize the rod configuration (discharge current of each rod) 
to maximize measured performance(s) of PMM.

We use CMA-ES (Covariance Matrix Adaptation Evolution Strategy), which is a
gradient-free method that MINIMIZES a given cost. For a "maximize" objective,
we return cost = -performance.

Steps:
 1) "ask" a batch of candidate solutions from CMA-ES
 2) measure them in the experiment (or fake environment)
 3) "tell" CMA-ES the fitness (cost) values
 4) let CMA-ES update its search distribution

If you want to test with the fake data code, just pick FakePMMInSitu
If you want the *real device*, pick PMMInSitu.

Dependencies:
  - pip install cma: https://github.com/CMA-ES/pycma
  - from PMMInSitu import PMMInSitu
  - or from FakePMMInSitu import FakePMMInSitu
  - run_cma_optimization(pm)
  - Possibly define your measure_cost function for your real or fake environment.
"""

import time
import numpy as np
import cma  # pip install cma

# Remember: use PMM environment, check folder
# TO PICK FAKE DATA SIM:
#   1) comment/uncomment the import lines
#   2) also see "def main()" at the bottom

# --- PICK ONE ---
# from PMMInSitu import PMMInSitu    # for the *real hardware* experiment
from FakePMMInSitu import FakePMMInSitu  # for *simulated* environment


# ------------------------------------------------------------------
# Global Settings (tweak as needed)
# ------------------------------------------------------------------
N = 100 # number of rods/parameters
INIT_SIGMA = 0.3 # initial step size (CMA default)
LOWER_BOUND = -1.0 # these bounds are chosen to include the ideal and also have enough range to explore and find the ideal. should always be zero amps irl
UPPER_BOUND =  1.0

# PMMInSitu waveguide objective knobs
FPM   = 7.0   # max plasma frequency in GHz for mapping
KNOB  = 0.5   # scaling knob
SCALE = 1.0   # more scaling
# If real hardware, you might vary these or pass them differently.

# ------------------------------------------------------------------
# measure_cost: returns a cost (lower=better), so if performance is better
# we do cost = -performance
# ------------------------------------------------------------------
def measure_cost(pm, rho):
    """
    CMA-ES Minimizes cost. If we want to MAXIMIZE some waveguide performance:
      cost = -performance

    Steps:
      1) Set rods in the device (or fake).
      2) Possibly measure S-parameters (if real device) or do something fake.
      3) Compute waveguide objective (the bigger the better).
      4) Return negative of that objective => cost.

    In real hardware:
      - pm.ArraySet_Rho(...) might physically set each rod.
      - Then measure performance from a sensor or S-parameters.
    """
    # 1) Set rods
    pm.ArraySet_Rho(rho, pm.f_a(FPM), knob=KNOB, scale=SCALE)
    time.sleep(0.05)  # small wait (fake or real hardware settle)

    # 2) If real device, measure S-params or do something. 
    #call "fake" S-parameters or do a trivial function in FakePMMInSitu
    freq, S21, S31 = pm.Get_S21_S31()  
    pm.Deactivate_Bulb('all')
    time.sleep(0.05)

    # 3) Suppose we do waveguide objective from pm.Wvg_Obj_Get(...)
    performance, _ = pm.Wvg_Obj_Get(
        rho,
        fpm = FPM,
        k   = KNOB,
        S   = SCALE,
        f   = 5.0,     # target freq
        df  = 0.25,    # bandwidth
        objective='dB',
        norms=[],
        duty_cycle=0.5
    )
    
    # 4) cost = -performance 
    cost = -performance
    return cost

# ------------------------------------------------------------------
# run_cma_optimization: main CMA-ES loop
# ------------------------------------------------------------------
def run_cma_optimization(pm):
    """
    Runs CMA-ES with a random initial guess in [LOWER_BOUND,UPPER_BOUND].
    The code picks solutions, we measure cost for each, then updates CMA-ES.

    'pm' can be an actual PMMInSitu or a FakePMMInSitu for testing.
    """
    # 1) initial guess
    x0 = np.random.uniform(LOWER_BOUND, UPPER_BOUND, N)

    # 2) CMA-ES options
    cma_opts = {
        "bounds": [LOWER_BOUND, UPPER_BOUND],  # clamp solutions
        "popsize": 2*N,                        # bigger pop for 100 dims
        "CMA_diagonal": True,                 # reduce complexity
        "maxiter": 50,                        # limit iterations
        "verb_disp": 1,                       # verbosity
    }

    # 3) create the CMA-ES instance
    es = cma.CMAEvolutionStrategy(x0, INIT_SIGMA, cma_opts)

    # 4) main CMA loop: ask -> measure_cost -> tell
    while not es.stop():
        solutions = es.ask()   # candidate solutions
        costs = []
        for sol in solutions:
            c = measure_cost(pm, sol)
            costs.append(c)
        es.tell(solutions, costs)

        # optional print
        print("[CMA-ES] Current best solution so far (internal CMA estimate):")
        es.disp()

    # 5) retrieve final results
    result = es.result
    best_rho  = result.xbest  # best solution found
    best_cost = result.fbest  # best cost
    # cost = -performance => performance = -best_cost
    best_performance = -best_cost

    print("\nCMA-ES finished.")
    print(f"Best cost found: {best_cost:.4f} => implies performance ~ {best_performance:.4f}")
    # print params with 4 decimals, no scientific notation
    with np.printoptions(precision=4, suppress=True):
        print("Best parameters (rho):", best_rho)

    # set device to best solution
    pm.ArraySet_Rho(best_rho, pm.f_a(FPM), knob=KNOB, scale=SCALE)
    time.sleep(0.1)
    print("[CMA-ES] Device set to best CMA-ES solution.")

# ------------------------------------------------------------------
# main script
# ------------------------------------------------------------------
def main():
    # Record the start time
    start_time = time.time()

    # pick PMM device
    # If real device: from PMMInSitu import PMMInSitu
    # pm = PMMInSitu("./../confs/conf_test.yaml")
    # or if fake:
    pm = FakePMMInSitu()
    print("[MAIN] Initialized FakePMMInSitu. Possibly warm up if needed.")
    pm.Config_Warmup(T=2)

    # run CMA-ES
    run_cma_optimization(pm)

    # measure total runtime
    end_time = time.time()
    elapsed = end_time - start_time
    hh = int(elapsed // 3600)
    mm = int((elapsed % 3600) // 60)
    ss = int(elapsed % 60)
    print(f"[MAIN] Total run time: {hh}h:{mm}m:{ss}s")

if __name__ == "__main__":
    main()
