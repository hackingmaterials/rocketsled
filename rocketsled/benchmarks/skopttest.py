from skopt.benchmarks import branin, hart6
from skopt import gbrt_minimize, forest_minimize, gp_minimize, dummy_minimize
import numpy as np
import pandas as pd
import math

def rastrigin(X):
    return 10*len(X) + sum([(x**2 - 10 * np.cos(2 * math.pi * x)) for x in X])

branindim = [(-5.0, 10.0), (0.0, 15.0)]  # branin
def hartdim(dim):
    return [(0, 1)] * dim
def rastrigindim (dim):
    return [(-5.12, 5.12)] * dim

OPTS = gbrt_minimize, forest_minimize, gp_minimize
FUNS = [rastrigin, branin, hart6]
DIMS = [branindim, hartdim(6), rastrigindim(15)]
NRUNS = 100
NPTS = 10000


if __name__ == "__main__":
    calls = 0


    vals = []
    for run in range(NRUNS):
        print("run {}".format(run))
        # res  = forest_minimize(branin, branindim, n_calls=50, n_jobs=4, n_points=NPTS)
        res = dummy_minimize(branin, branindim, n_calls=50)
        vals.append(res.fun)
        print res.fun
        print res.x

    print "MEAN IS: ", np.mean(vals)