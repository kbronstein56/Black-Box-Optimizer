"""
CMA-ES-GI referencing ideas from
    Jin Jin, Chuan Yang, Yi Zhang, 
    "An Improved CMA-ES for Solving Large Scale Optimization Problem," 
    in Advances in Swarm Intelligence, LNCS 12145, 2020.

With:
  - Standard CMA-ES rank-based update for mean and covariance,
  - "Gradient Information" from the entire population (search gradient adaptation),
  - Weighted 'score' for each sample that helps shape a search gradient,
  - A specialized partial derivative approach to update distribution parameters 
    (somewhat similar to eq. (5) or eq. (6) from [1]).

Use:
  - from FakePMMInSitu import FakePMMInSitu  (or PMMInSitu for real hardware)
  - Then define waveguide measure function. 
  - Adjust population size, learning rates, and GI blending as you like.
"""

import numpy as np
import math
import time
import random

# Swap if you want real hardware:
# from PMMInSitu import PMMInSitu
from FakePMMInSitu import FakePMMInSitu

# ==============================================================
# Global waveguide / rods settings
# ==============================================================
N_DIM = 100
LOWER_BOUND = -1.0
UPPER_BOUND =  1.0

POP_SIZE = 60        # λ, total offspring
MU = 30              # number of top parents
MAX_ITER = 50
INIT_RADIUS = 0.3    # akin to sigma in CMA
GI_BLEND = 0.1       # how strongly we blend in GI updates

# waveguide objective knobs
FPM   = 7.0  
KNOB  = 0.5  
SCALE = 1.0  
F_TARG= 5.0  
DF_BW = 0.25 
DUTY  = 0.5  

# ==============================================================
# measure performance from PMM
# ==============================================================
def measure_waveguide_performance(pm, x):
    """
    Sets rods to x, measure waveguide performance from pm,
    returns a float (the bigger, the better).
    """
    pm.ArraySet_Rho(x, pm.f_a(FPM), knob=KNOB, scale=SCALE)
    time.sleep(0.01)

    perf, _ = pm.Wvg_Obj_Get(
        x,
        fpm=FPM,
        k=KNOB,
        S=SCALE,
        f=F_TARG,
        df=DF_BW,
        objective='dB',
        norms=[],
        duty_cycle=DUTY
    )
    pm.Deactivate_Bulb('all')
    time.sleep(0.01)
    return perf

# ==============================================================
# GI-CMAES Class
# ==============================================================
class GICMAES_Advanced:
    """
    A more advanced CMA-ES variant that attempts to incorporate 
    the 'Gradient Information' approach described in [1].
    This includes:
      - Weighted rank-based updates for mean/cov
      - A GI approach that uses the entire population to compute a 
        "search gradient" that modifies the distribution updates.
    """

    def __init__(self, dimension, measure_func,
                 lower_bound, upper_bound,
                 population_size=POP_SIZE,
                 mu=MU,
                 max_iter=MAX_ITER,
                 init_radius=INIT_RADIUS,
                 gi_blend=GI_BLEND):
        self.dimension = dimension
        self.measure_func = measure_func
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.population_size = population_size  # λ
        self.mu = mu                             # top parents
        self.max_iter = max_iter
        self.radius = init_radius               # akin to CMA sigma
        self.gi_blend = gi_blend                # how strongly to blend GI

        # init mean
        self.mean = np.random.uniform(self.lower_bound, self.upper_bound, self.dimension)
        # init covariance
        self.cov = np.identity(self.dimension)
        self.B = np.identity(self.dimension)
        self.D = np.ones(self.dimension)

        # standard CMA learning rates
        # rank-based
        # typically c1, cmu ~ 2/(dim+1.3)^2, etc
        self.c1 = 2.0 / ((self.dimension + 1.3)**2 + self.mu)
        self.cmu= min(1.0 - self.c1,
                      2.0*(self.mu-2.0 + 1.0/self.mu)/((self.dimension+2.0)**2 + self.mu))
        # step-size path stuff
        self.cs = (self.mu+2)/(self.dimension+self.mu+5)
        self.cc = (4 + self.mu/self.dimension)/(self.dimension+4+2*self.mu/self.dimension)
        self.damps = 1.0 + 2.0*max(0, math.sqrt((self.mu-1)/(self.dimension+1))-1) + self.cs

        # rank-based weights
        log_seq = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu+1))
        self.weights = log_seq / np.sum(log_seq)
        self.mu_eff = 1.0 / np.sum(self.weights**2)

        # evolution paths
        self.path_s = np.zeros(self.dimension)
        self.path_c = np.zeros(self.dimension)

        # store best
        self.best_perf = -1e10
        self.best_x = self.mean.copy()
        self.current_iter = 0

    def clamp(self, x):
        return np.minimum(np.maximum(x, self.lower_bound), self.upper_bound)

    def update_eig(self):
        # recalc B,D from cov
        eigvals, eigvecs = np.linalg.eigh(self.cov)
        eigvals = np.maximum(eigvals, 1e-30)
        self.B = eigvecs
        self.D = np.sqrt(eigvals)

    def sample_population(self):
        """
        x = mean + radius * B*(D*z)
        """
        pop = []
        for _ in range(self.population_size):
            z = np.random.randn(self.dimension)
            y = self.B.dot(self.D*z)
            x = self.mean + self.radius*y
            x = self.clamp(x)
            pop.append(x)
        return np.array(pop)

    def approximate_gi(self, population, fitness):
        """
        More advanced GI approach, referencing eq. (5)/(6) style logic in [1].
        We'll do:
          1) define a 'score' for each sample i, e.g. score_i = (f_i - f_avg) 
             or a normalized version
          2) Then define a "gradient" ~ sum( score_i * [ (z_i)(z_i)^T - I ] ) 
             or something similar, as in eq. (5). 
        We'll produce a partial derivative wrt the distribution parameters.

        For demonstration, we do a partial approach:
          Let z_i = B^T (x_i - mean)/radius / D
          Then eq. (5) in [1] uses sum_i score_i (z_i z_i^T - I)
          That modifies e.g. the distribution in some rank-mu sense.

        We'll return a matrix G that we can incorporate into cov update, 
        and a vector g_mean for mean update, for example.
        """
        # 1) compute z_i for each sample
        pop_size = len(population)
        X_minus_mean = population - self.mean
        # We also compute z_i in the CMA sense
        Z = []
        for i in range(pop_size):
            z_i = np.linalg.solve(np.diag(self.D), self.B.T.dot(X_minus_mean[i]/self.radius))
            Z.append(z_i)
        Z = np.array(Z)

        # 2) define score_i
        f_avg = np.mean(fitness)
        f_std = np.std(fitness) + 1e-9
        scores = (fitness - f_avg) / (f_std)  # normalized

        # 3) eq. (5)-like approach: sum_i scores_i [ z_i z_i^T - I ]
        # We'll produce a "Gcov" for covariance, and "Gmean" for mean
        # or incorporate a partial derivative for each
        Gcov = np.zeros((self.dimension, self.dimension))
        for i in range(pop_size):
            z_i = Z[i]
            outer_zz = np.outer(z_i, z_i)
            Gcov += scores[i]*(outer_zz - np.identity(self.dimension))
        Gcov /= pop_size

        # for the mean, we might do something akin to eq. (6) or a simpler approach:
        # GI can guide mean if we do partial derivatives wrt mean
        # we'll define Gmean ~ sum_i scores_i*(x_i - mean)
        Gmean = np.zeros(self.dimension)
        for i in range(pop_size):
            Gmean += scores[i]*X_minus_mean[i]
        Gmean /= pop_size

        return Gmean, Gcov

    def ask(self):
        return self.sample_population()

    def tell(self, population, performance):
        self.current_iter += 1
        # track best
        idx_best = np.argmax(performance)
        if performance[idx_best] > self.best_perf:
            self.best_perf = performance[idx_best]
            self.best_x = population[idx_best].copy()

        # rank-based new mean (like standard CMA)
        idx_sort = np.argsort(-performance)  # descending
        top_indices = idx_sort[:self.mu]
        new_mean = np.zeros(self.dimension)
        for i in range(self.mu):
            new_mean += self.weights[i]*population[top_indices[i]]

        # standard CMA path for step-size
        y = (new_mean - self.mean)/self.radius
        z = np.linalg.solve(np.diag(self.D), self.B.T.dot(y))

        cs_ = self.cs
        self.path_s = (1.0 - cs_)*self.path_s + math.sqrt(cs_*(2.0-cs_)*self.mu_eff())*z

        # check hsig
        ps_norm = np.linalg.norm(self.path_s)
        threshold = (1.4 + 2.0/(self.dimension+1.0))*self.chi_n()
        hsig = (ps_norm / math.sqrt(1.0 - (1.0 - cs_)**(2*self.current_iter))) < threshold

        # path for cov
        cc_ = self.cc
        self.path_c = (1.0 - cc_)*self.path_c
        if hsig:
            self.path_c += math.sqrt(cc_*(2.0-cc_)*self.mu_eff())*(y.dot(self.B)/self.D)
        # else no special correction here

        # rank-one update
        c1_ = self.c1
        cmu_ = self.cmu
        rank_one = np.outer(self.path_c, self.path_c)
        if not hsig:
            rank_one += cc_*(2.0-cc_)*self.cov

        self.cov *= (1.0 - c1_ - cmu_)
        self.cov += c1_*rank_one

        # rank-mu
        old_mean = self.mean
        for i in range(self.mu):
            diff = (population[top_indices[i]] - old_mean)/self.radius
            self.cov += cmu_*self.weights[i]*np.outer(diff,diff)

        # incorporate GI from entire population
        Gmean, Gcov = self.approximate_gi(population, performance)
        # blend it in
        self.mean = new_mean + self.gi_blend * Gmean
        # incorporate Gcov into self.cov in a small rank update
        # e.g. Cov <- Cov + alpha * B * Gcov * B^T, or directly in z-space
        # for simplicity, let's do a direct transform in "z" space approach
        # but we'll do a naive approach: Cov += gi_blend * radius^2 * B*(Gcov)*B^T
        # since Gcov is in z-coords
        alpha_cov = self.gi_blend*0.1
        # transform Gcov back to x-space: B*(Gcov)*B^T
        Gcov_xspace = self.B.dot(Gcov).dot(self.B.T)
        self.cov += alpha_cov*(self.radius**2)*Gcov_xspace

        # update radius (like standard CMA step-size adaptation)
        ds_ = self.damps
        self.radius *= math.exp( min(1.0, (cs_/ds_)*((ps_norm/self.chi_n())-1.0)) )

        # re-eigen
        self.update_eig()

    def chi_n(self):
        return math.sqrt(self.dimension)*(1.0 - 1.0/(4.0*self.dimension) + 1.0/(21.0*self.dimension**2))

    def mu_eff(self):
        return 1.0/np.sum(self.weights**2)

    def run(self):
        self.update_eig()
        for iteration in range(self.max_iter):
            pop = self.ask()
            perf = np.array([self.measure_func(x) for x in pop])
            self.tell(pop, perf)
            print(f"[GI-CMAES] Iter={self.current_iter}, best={self.best_perf:.4f}, radius={self.radius:.3f}")
        return self.best_x, self.best_perf

# ==============================================================
# main
# ==============================================================
def main():
    pm = FakePMMInSitu()  # or PMMInSitu("./conf.yaml")
    pm.Config_Warmup(T=2)

    start_time = time.time()

    def measure_perf(x):
        return measure_waveguide_performance(pm, x)

    gi_cma = GICMAES_Advanced(
        dimension=N_DIM,
        measure_func=measure_perf,
        lower_bound=LOWER_BOUND,
        upper_bound=UPPER_BOUND,
        population_size=POP_SIZE,
        mu=MU,
        max_iter=MAX_ITER,
        init_radius=INIT_RADIUS,
        gi_blend=GI_BLEND
    )

    best_x, best_perf = gi_cma.run()

    end_time = time.time()
    elapsed = end_time - start_time

    print("\n[GI-CMAES ADVANCED] Finished.")
    print(f"Best performance found = {best_perf:.4f}")
    print("Best rods array:")
    print(best_x)

    pm.ArraySet_Rho(best_x, pm.f_a(FPM), knob=KNOB, scale=SCALE)
    time.sleep(0.1)
    print("[GI-CMAES ADVANCED] Device set to best solution.")
    print(f"Total run time: {elapsed:.2f} s")

if __name__ == "__main__":
    main()
