#rembember: use PMM environment
#check folder

#TO PICK FAKE DATA SIM: 
    # 1. change import file at beginning 
    # 2. AND for def waveguide_performance 
    # 3. AND for def run_ga_optimization 
    # 4. AND change main at very end! 
    # this will call fake class's "ArraySet_Rho" and Wvg_Object_Get --> see if converges near .3

# Genetic Algorithm for In-Situ PMM Optimization
"""
The idea is to optimize the rod configuration (discharge current of each rod) 
to maximize measured performance(s) of PMM.
Will measure S-parameters from VNA...so like S_21 might represent transmission from port 1 to 2.
S is power transmitted (in dB?)
Then compute scalar objective (like maximize power in port 2 and minimize leakage in port 3).
pm.Wvg_Obj_Get() is an example that tries to make sure that energy goes mostly to output port 2 at the chosen frequency.

This is a GA that tunes a 1D array of rho parameters (one per rod). 
Each individual in the GA is a vector of length N (number of rods). 
The GA will iteratively:
1) Evaluate the population on the plasma device with PMMInSitu.
2) Select parents.
3) Apply crossover and mutation to create offspring.
4) Form the next generation.

So first defines a waveguide_performance() function that sets the
rods (rho parameters) in real experiment, measures performance,
and returns a scalar (higher is better).
Then initializes a population of random solutions, evaluates them,
and iteratively select, crossover, mutate, and replace the population
for some number of generations.

Will modify:
 - measure_fitness() to call the correct PMMInSitu objective function  (like waveguide or demux)
 - the parameter bounds (INIT_MIN, INIT_MAX) as needed
 - the number of rods N, population size POP_SIZE, etc...
 - TO PICK FAKE DATA SIM: change import file at beginning and change main at very end! 
    this will call fake class's "ArraySet_Rho" and Wvg_Object_Get --> see if converges near 0.3

Dependencies:
 - PMMInSitu.py in same folder
 - conf_test.yaml
 
Use:
  1. from PMMInSitu import PMMInSitu
  2. pm = PMMInSitu(...)

call run_ga_optimization(pm) to run the GA.

"""

import random
import time
import numpy as np

# Import experimental class: PICK between real and fake data
#from PMMInSitu import PMMInSitu #real PMM experiment
from FakePMMInSitu import FakePMMInSitu #FOR SIMULATED FAKE DATA


# Settings (remember to update as needed!)
NUM_RODS = 100 # number of rods/parameters
POPULATION_SIZE = 100 # population size in GA
MAX_GENERATIONS = 50 # number of generations
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1
INIT_MIN = -1.0 # initial random lower bound....idk what my my bounds should be??
INIT_MAX =  1.0 # initial random upper bound

# PMMInSitu waveguide objective knobs
FPM = 7.0 # max plasma frequency in GHz for mappin
KNOB = 0.5 # like..choose a lower/higher range of currents...
SCALE = 1.0 # ...like multiply final freq of current level up/down...knob and scale: to scale the plasma-frequency to current mapping...adjustment for going from mapping to actual power source settings
F_TARG = 5.0 # waveguide target freq (GHz)
DF_BW = 0.25 # waveguide bandwidth
DUTY = 0.5 # used in wait times, see PMM code
# rho # parameter (aka frequency)... i-th rod's tunning parameter that are interpreted by knob and scale when set actual ps currents

# ---------------------------------------------------------
def waveguide_performance(pm: FakePMMInSitu, rho: np.ndarray) -> float: ##CHANGE PMMInSitu for real exp
# ---------------------------------------------------------
    # 1) physically set rods (to rho)
    pm.ArraySet_Rho(rho, pm.f_a(FPM), knob=KNOB, scale=SCALE)
    time.sleep(0.01)  # let hardware settle (plasma rods need cooled? coils need cooled?)

    # 2) waveguide objective - measure performance with pm.Wvg_Obj_Get(...)
    performance, _ = pm.Wvg_Obj_Get(
        rho,
        fpm = FPM,
        k = KNOB,
        S = SCALE,
        f = F_TARG,
        df = DF_BW,
        objective = 'dB',
        norms = [],
        duty_cycle = DUTY
    )

    # Deactivate bulbs after measurement
    pm.Deactivate_Bulb('all')
    time.sleep(0.01)
    
    # 3) Return a scalar performance (higher = better)
    return performance

# ----------------------------------------------------
# Operators: individual, crossover, mutate, tournament
# ----------------------------------------------------

def create_individual() -> np.ndarray:
    """Random array of length NUM_RODS in [INIT_MIN, INIT_MAX]."""
    return np.array([
        random.uniform(INIT_MIN, INIT_MAX)
        for _ in range(NUM_RODS)
    ])

def crossover(parent1: np.ndarray, parent2: np.ndarray):
    """Blend crossover with probability CROSSOVER_RATE.
    For example a child could get half of one parents dna and half of the others."""
    c1, c2 = parent1.copy(), parent2.copy()
    if random.random() < CROSSOVER_RATE:
        alpha = random.random()
        for i in range(NUM_RODS):
            c1[i] = alpha * parent1[i] + (1 - alpha) * parent2[i]
            c2[i] = alpha * parent2[i] + (1 - alpha) * parent1[i]
    return c1, c2

def mutate(ind: np.ndarray):
    """Mutate each gene with probability MUTATION_RATE (Gaussian noise).
    Retains genetic diversity so doesn't converge too quickly.
    First pick each gene to mutate with some small probability like .01
    Then if chosen, add some noise to that rod's parameter.
    Then clamp if goes out of the range.
    So, each child can deviate slightly from parents and find new solutions that the parent didn't have."""
    for i in range(NUM_RODS):
        if random.random() < MUTATION_RATE:
            noise = random.gauss(0.0, 0.01)  # e.g. sigma=0.05
            ind[i] += noise
            # optional clamp:
            if ind[i] < INIT_MIN:
                ind[i] = INIT_MIN
            if ind[i] > INIT_MAX:
                ind[i] = INIT_MAX

"""def mutate_extreme(ind, probability=0.05):
    # With a small probability, do a big random jump
    if random.random() < probability:
        i = random.randint(0, len(ind)-1)
        ind[i] = random.uniform(INIT_MIN, INIT_MAX)"""
def mutate_extreme(ind, extreme_sigma=1.5, fraction=1.0):
    """
    Apply a bigger Gaussian noise (og sigma=0.2) to 'ind'.
    'fraction' is how many genes to mutate proportionally.

    :param ind: The array of rod parameters (1D numpy array).
    :param extreme_sigma: The std dev for the large mutation.
    :param fraction: The fraction of genes to mutate. (0.3 => 30% of genes get jolted)
    """
    num_genes = len(ind)
    num_to_mutate = int(num_genes * fraction)
    # randomly pick which indices to mutate
    mutate_indices = random.sample(range(num_genes), num_to_mutate)

    for i in mutate_indices:
        # big random jump
        noise = random.gauss(0.0, extreme_sigma)
        ind[i] += noise
        # clamp to [INIT_MIN, INIT_MAX]
        if ind[i] < INIT_MIN:
            ind[i] = INIT_MIN
        elif ind[i] > INIT_MAX:
            ind[i] = INIT_MAX


def select_parent(pop, fitnesses):
    """Tournament selection of size 3.
    First randomly pick 3 individuals from population.
    Then compare their fitness.
    Then the one with the best fitness wins and is chosen as a parent.
    So, the better performing ones are more likely to reproduce, but still some rondomness so dont get stuck too early."""
    contenders = random.sample(range(len(pop)), 3)
    best_idx = contenders[0]
    for c in contenders[1:]:
        if fitnesses[c] > fitnesses[best_idx]:
            best_idx = c
    return pop[best_idx]

# ----------------------------------------------
# MAIN GA LOOP
# ----------------------------------------------
def run_ga_optimization(pm: FakePMMInSitu): ##CHANGE PMMInSitu for real exp!!!
    
    # 1) start population
    population = [create_individual() for _ in range(POPULATION_SIZE)]
    fitnesses  = []

    # 2) Evaluate initial population
    for ind in population:
        score = waveguide_performance(pm, ind)
        fitnesses.append(score)

    best_score = max(fitnesses)
    best_ind   = population[np.argmax(fitnesses)].copy()
    print(f"[GA] Initial best fitness: {best_score:.4f}")

    # 3) evolve
    for gen in range(MAX_GENERATIONS):
        new_population = []
        new_fitnesses  = []

        # produce children in pairs
        while len(new_population) < POPULATION_SIZE:
            p1 = select_parent(population, fitnesses)
            p2 = select_parent(population, fitnesses)
            c1, c2 = crossover(p1, p2)
            mutate(c1)
            mutate(c2)

            # measure children
            f1 = waveguide_performance(pm, c1)
            f2 = waveguide_performance(pm, c2)

            new_population.append(c1)
            new_population.append(c2)
            new_fitnesses.append(f1)
            new_fitnesses.append(f2)

        # trim if overshoot
        new_population = new_population[:POPULATION_SIZE]
        new_fitnesses = new_fitnesses[:POPULATION_SIZE]

        #check if we improved
        gen_best = max(new_fitnesses)
        if gen_best > best_score:
            best_score = gen_best
            best_ind = new_population[np.argmax(new_fitnesses)].copy()
            stall_counter = 0  # reset stall if improved
        else:
            stall_counter += 1

        # -- JOLT if we are stuck --
        STALL_LIMIT = 2
        if stall_counter >= STALL_LIMIT:
            print("[GA] Stall detected. Doing an EXTREME MUTATION on the population!")
            for ind_idx in range(POPULATION_SIZE):
                # either do it for entire population
                mutate_extreme(new_population[ind_idx], extreme_sigma=0.2, fraction=1.0)
            # Re-evaluate after the jolt
            new_fitnesses = []
            for ind in new_population:
                f = waveguide_performance(pm, ind)
                new_fitnesses.append(f)
            # check for improvement after jolt
            gen_best = max(new_fitnesses)
            if gen_best > best_score:
                best_score = gen_best
                best_ind = new_population[np.argmax(new_fitnesses)].copy()
            stall_counter = 0  # reset again

        #update population for next generation
        population = new_population
        fitnesses = new_fitnesses
        
        print(f"[GA] Generation {gen+1}/{MAX_GENERATIONS}, best so far = {best_score:.4f}")

    print("[GA] Done. Best fitness =", best_score)
    #print("[GA] Best individual (rho) =", best_ind)
    with np.printoptions(precision=4, suppress=True):
        print("[GA] Best individual rho =", best_ind)
    # set rods to best at end
    pm.ArraySet_Rho(best_ind, pm.f_a(FPM), knob=KNOB, scale=SCALE)
    time.sleep(0.01)

    print("[GA] Final rods set to best solution.")


# --------------------------------
# main script
# --------------------------------
def main():
    # Record the start time
    start_time = time.time()
    
    # 1) use PMMInSitu
    #pm = PMMInSitu("./../confs/conf_test.yaml")  # adapt path
    #print("Initialized PMMInSitu. Possibly do warmup:")
    #pm.Config_Warmup(T=2, ballasts='New', duty_cycle=DUTY)
    
    #OR USE FAKE DATA
    pm = FakePMMInSitu() # no config file needed this time
    print ("Initialized FakePMMInSitu. Warmup for demonstration:")
    pm.Config_Warmup(T=2)

    # 2) Run GA
    run_ga_optimization(pm)
    
    # Record the end time
    end_time = time.time()

    # Calculate elapsed time in seconds
    elapsed_seconds = end_time - start_time
    
    # break it down into hours, mins, secs
    hours = int(elapsed_seconds // 3600)
    mins = int((elapsed_seconds % 3600) // 60)
    secs = int(elapsed_seconds % 60)
    
    print(f"Total run time: {hours}h:{mins}m:{secs}s")

if __name__ == "__main__":
    main()
