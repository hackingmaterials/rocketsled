import time
import warnings
import datetime
import numpy as np
from matplotlib import pyplot as plt
from skopt.benchmarks import branin, hart6
from scipy.optimize import rosen
from rocketsled import auto_setup
from rocketsled.utils import Dtypes
import random
from fireworks import LaunchPad
from itertools import product
import pandas as pd
import math

dtypes = Dtypes()

def visualize(csets, maximize, labels, colors, limit=0, fontfamily="serif"):
    opt = max if maximize else min

    for l, cset in enumerate(csets):
        bestbig = []
        for c in cset:
            fx = []
            i = []
            best = []
            for doc in c.find({'index': {'$exists': 1}}).limit(limit):
                fx.append(doc['y'])
                i.append(doc['index'])
                best.append(opt(fx))
            bestbig.append(best)

        for b in bestbig:
            print(b)

        mean = np.mean(bestbig, axis=0)
        std = np.std(bestbig, axis=0)
        avgbest = opt(mean)
        print(avgbest)
        plt.plot(i, mean, label=" Avg best {}: {}".format(labels[l], avgbest), color=colors[l])
        plt.fill_between(i, mean - std, mean + std, color=colors[l], alpha=0.2)

    plt.rc('font', family=fontfamily)
    #plt.yscale("log")
    plt.xlabel("fx evaluation")
    plt.ylabel("fx value")
    plt.legend()
    return plt


def ran_run(func, dims, runs, comps_per_run, maximize):
    opt = max if maximize else min
    best = np.zeros((runs, comps_per_run))
    for r in range(runs):
        print("RUN {} of {}".format(r, runs))
        rany = np.zeros(comps_per_run)
        ranx = np.asarray([np.random.uniform(low=d[0], high=d[1], size=comps_per_run) for d in dims]).T
        for c in range(comps_per_run):
            rany[c] = func(ranx[c])
        for c in range(1, comps_per_run):
            rany[c] = opt([rany[c], rany[c-1]])
        best[r] = rany
    meanbest = np.mean(best, axis=0)
    return ([i + 1 for i in range(comps_per_run)], meanbest)

def ds(dims):
    total_dimspace = []
    for dim in dims:
        if len(dim) == 2:
            lower = dim[0]
            upper = dim[1]
            if type(lower) in dtypes.ints:
                # Then the dimension is of the form (lower, upper)
                dimspace = list(range(lower, upper + 1))
            elif type(lower) in dtypes.floats:
                    dimspace = [random.uniform(lower, upper) for _ in range(1000)]
            else:  # The dimension is a discrete finite string list of two entries
                dimspace = dim
        else:  # the dimension is a list of categories or discrete integer/float entries
            dimspace = dim
        random.shuffle(dimspace)
        total_dimspace.append(dimspace)
    space = [[xi] for xi in total_dimspace[0]] if len(dims) == 1 else product(*total_dimspace)
    return space


def rastrigin(X):
    return 10*len(X) + sum([(x**2 - 10 * np.cos(2 * math.pi * x)) for x in X])

def rastrigindim (dim):
    return [(-5.12, 5.12)] * dim


def hartdim():
    return [(0.0, 1.0)]*6


if __name__ == "__main__":
    dim = [(-5.0, 10.0), (0.0, 15.0)] # branin

    # GENERATE PYTHON FILES
#    for i in range(100):
#         auto_setup(rastrigin, rastrigindim(30), wfname='ptrast{}'.format(i), opt_label='ei{}'.format(i), host='localhost', acq='ei', name="ptrast", port=27017, n_boots=1000, predictor="RandomForestRegressor", n_searchpts=2000)

    # GENERATE NEW RANDOM FILE
    ranx, rany = ran_run(rastrigin, rastrigindim(30), runs=1000, comps_per_run=50, maximize=False)
    df = pd.DataFrame({'x': ranx, 'y': rany}).to_csv("ALEXGRAPHER_ran_rast.csv")
    
    # LOAD AND GRAPH
    lpad = LaunchPad(host='localhost', port=27017, name='ptrast')
    df = pd.DataFrame.from_csv("ALEXGRAPHER_ran_rast.csv")
    ranx = df['x']
    rany = df['y']
    ei_runs = [getattr(lpad.db, "ei{}".format(i))for i in range(100)]
#    lpad_highest = LaunchPad(host='localhost', port=27017, name='ptbranin_highest')
#    ei_runs_highest = [getattr(lpad_highest.db, "ei{}".format(i)) for i in range(100)]
#    plt = visualize([ei_runs, ei_runs_highest], False, labels=['EI', 'highest'], colors=['blue', 'green'], limit=50)
    plt = visualize([ei_runs], False, labels=['EI'], colors=['blue'], limit=30)
    plt.plot(ranx, rany, color='black')
    print(min(rany))
    plt.show()
