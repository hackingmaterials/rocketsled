import time
import warnings
import datetime
import numpy as np
from matplotlib import pyplot as plt
from skopt.benchmarks import branin, hart6
from rocketsled import auto_setup
from rocketsled.utils import Dtypes
from scipy.stats import f_oneway
import random
from fireworks import LaunchPad
from itertools import product
import pandas as pd
import math


"""
READ THIS FIRST:
You should install skopt. pip install scikit-optimize. It's not needed but
might help resolve annoying import errors.

The built-in optimizers can't really be used outside of rocketsled (unless
you rewrite parts of them, which can be done but is a pain). Since benchmarking
functions are usually just python functions (not complex/expensive workflows),
we can use rocketsled auto_setup to create workflows to evaluate their
performance. For tests so far I have used 50 allowed function evaluations per 
test and 100 runs for each optimizer on each benchmark function. Scroll down
for instructions.

For non-rocketsled optimizers (i.e., for comparison) you really DO NOT NEED this
file except for helping with visualization. Using optimization-only libraries to 
benchmark these functions should be super easy and not require any of this. 
"""

dtypes = Dtypes()

def visualize(csets, opt, labels, colors, fontfamily="serif", limit=0):
    """
    Visualize a built-in optimizer's performance by reading the set of mongo
    collections, analyzing data, and plotting in matplotlib.

    Args:
        csets:
        opt:
        labels:
        colors:
        fontfamily:
        limit:

    Returns:

    """
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

        # for b in bestbig:
        #     print(b)

        mean = np.mean(bestbig, axis=0)
        std = np.std(bestbig, axis=0)
        avgbest = opt(mean)
        # print(avgbest)
        plt.plot(i, mean, label="RS RF", color=colors[l])
        # plt.fill_between(i, mean - std, mean + std, color=colors[l], alpha=0.2)

    plt.rc('font', family=fontfamily)
    fig = plt.gcf()
    fig.set_size_inches(4, 3)
    # plt.yscale("log")
    plt.xlabel("f(x) evaluation")
    plt.ylabel("minimum f(x) value")
    plt.title("Hartmann 6D Function")
    plt.legend()
    return (mean[-1], std[-1])


def gather_stats(cset, opt, limit=0):
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
    return [bb[-1] for bb in bestbig]

def ran_run(func, dims, opt, runs, comps_per_run, stats=False):
    """
    Run a function on a new function. You shouldn't run this more than 1x per
    obj function unless you need extra stats. Just save the results in a csv

    Args:
        func: The function handle
        dims: The dimensions, in rocketsled format (ie, bounded)
        opt: either max or min
        runs: number of runs to repeat (for statistical significance)
        comps_per_run: Number of allowed obj func evals per run
        stats: Return all the vals if true, else just the

    Returns:
        Either all runs data or just mean data depending on stats daa
    """
    best = np.zeros((runs, comps_per_run))
    for r in range(runs):
        print("RUN {} of {}".format(r, runs))
        rany = np.zeros(comps_per_run)
        ranx = np.transpose([np.random.uniform(low=dim[0], high=dim[1], size=comps_per_run) for dim in dims])
        for c in range(comps_per_run):
            rany[c] = func(ranx[c])
        for c in range(1, comps_per_run):
            rany[c] = opt([rany[c], rany[c-1]])
        best[r] = rany

    if stats:
        return best[:, -1]
    else:
        meanbest = np.mean(best, axis=0)
        return ([i + 1 for i in range(comps_per_run)], meanbest)


def rast(X):
    """
    Rastrigin functon - Extensible to N dimensions
    """
    return 10*len(X) + sum([(x**2 - 10 * np.cos(2 * math.pi * x)) for x in X])

#
def rastdim (dim):
    """
    Returns dimensions used for rastrigin function
    """
    return [(-5.12, 5.12)] * dim

#
def hartdim(dim):
    """
    Hartmann dimensions (use 3D or 6D)
    """
    return [(0.0, 1.0)] * dim

#
def rose(x):
    """
    Rosenbrock function - Extensible to N dimensions, but we using 2
    """
    x = np.asarray(x)
    r = np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0, axis=0)
    return r

def rosedim(dim):
    """
    Rosenbrock function dimensions
    """
    return [(-5.0, 10.0)] * dim


def schaffer(x):
    """
    Schaffer F4 (I think) function, we didn't use taht one very much
    """
    return 0.5 + ((math.sin(x[0]**2 - x[1]**2))**2 - 0.5)/\
           ((1.0 + .001 * (x[0]**2 + x[1]**2)) ** 2)


if __name__ == "__main__":


    # For julien - An example of running a full GP benchmark 100 times on the Branin-Hoo function

    # define dimensions of the search space (all bounded in this case)
    branindim = [(-5.0, 10.0), (0.0, 15.0)]

    # define the number of full GP benchmarks we want to repeat
    n_runs = 100
    db_name = 'bran_gp'

    for i in range(n_runs):
        # Set all your run options here. See the available options in the OptTask doc or the online comprehensive guide
        auto_setup(branin, branindim, wfname='bran_gp{}'.format(i),
                   opt_label='ei{}'.format(i), host='localhost', acq='ei',
                   name=db_name, port=27017, n_bootstraps=1000,
                   predictor="GaussianProcessRegressor", n_search_points=10000)

    # Now its the time to actually run everything. Open a terminal in this directory
    # > cenv3                                   # whatever virtualenv command you use
    # > lpad init                               # use all default options, except for "name", which should be bran_gp. This creates the launchpad file for the db which holds all optimization data
    # > lpad -l bran_gp.yaml reset              # press Y if warned, this just initializes the database
    # > cd ../auto_sleds
    # > for f in bran_gp*; do python $f; done   # add all our workflows to the launchpad
    # > cd ../benchmarks
    # We have 2 fireworks in each workflow (one evaluating f(x) and one optimization). And we have 50 optimization loops
    # to run for each of 100 runs. So there are 10,000 workflows to be run, total. To run them,
    # > rlaunch -l bran_gp.yaml multi 4 --nlaunches 2525
    # This command runs 4 parallel processes each launching 2500 fireworks. In total, this is 10100 fireworks.

    # Load results for random Branin optimization
    # df = pd.DataFrame.read_csv("ran_bran.csv")
    # ranx = df['x']
    # rany = df['y']

    # Plot 
    # lpad = LaunchPad(host='localhost', port=27017, name=db_name)
    # bm, bs = visualize([ei_runs], min, labels=['EI'], colors=['blue'], limit=50)
    # plt.plot(ranx, rany, color='black', label="Random")
    # plt.tight_layout()
    # plt.legend()
    # print("BEST RANDOM", min(rany))
    # print("BEST OPT", bm, "+-", bs)




    ##### Code to reproduce current graphs is below (don't use) ######

    # BRANIN 2D RF
    # dim = [(-5.0, 10.0), (0.0, 15.0)] # branin
    # # for i in range(100):
    # #     auto_setup(branin, dim, wfname='bran{}'.format(i), opt_label='ei{}'.format(i), host='localhost', acq='ei', name='bran', port=27017, n_bootstraps=1000, predictor="RandomForestRegressor", n_search_points=10000)
    #
    # # ranx, rany = ran_run(branin, dim, min, runs=10000, comps_per_run=50)
    # pd.DataFrame({'x': ranx, 'y': rany}).to_csv("ran_bran.csv")
    # lpad = LaunchPad(host='localhost', port=27017, name='bran')
    # df = pd.DataFrame.from_csv("ran_bran.csv")
    # ranx = df['x']
    # rany = df['y']
    # ei_runs = [getattr(lpad.db, "ei{}".format(i)) for i in range(100)]
    # bm, bs = visualize([ei_runs], min, labels=['EI'], colors=['blue'], limit=50)
    # plt.plot(ranx, rany, color='black', label="Random")
    # plt.tight_layout()
    # plt.legend()
    # print "BEST RANDOM", min(rany)
    # print "BEST OPT", bm, "+-", bs
    # rans = ran_run(branin, dim, min, runs=100000, comps_per_run=50, stats=True)
    # best = visualize2(ei_runs, min)
    # print("ONE WAY ANOVA:", f_oneway(rans, best))
    # plt.savefig("branin.png", dpi=200)
    # plt.show()


    # ROSEN 2D RF
    # dim = rosedim(2)
    # for i in range(100):
    #     auto_setup(rose, dim, wfname='rose{}'.format(i), opt_label='ei{}'.format(i), host='localhost', acq='ei', name='rose', port=27017, n_bootstraps=1000, predictor="RandomForestRegressor", n_search_points=10000)
    #
    # ranx, rany = ran_run(rose, dim, min, runs=10000, comps_per_run=50)
    # pd.DataFrame({'x': ranx, 'y': rany}).to_csv("ran_rose.csv")
    # lpad = LaunchPad(host='localhost', port=27017, name='rose')
    # df = pd.DataFrame.from_csv("ran_rose.csv")
    # ranx = df['x']
    # rany = df['y']
    # ei_runs = [getattr(lpad.db, "ei{}".format(i)) for i in range(100)]
    # bm, bs = visualize([ei_runs], min, labels=['EI'], colors=['blue'], limit=50)
    # plt.plot(ranx, rany, color='black', label="Random")
    # plt.tight_layout()
    # plt.legend()
    # print "BEST RANDOM", min(rany)
    # print "BEST OPT", bm, "+-", bs

    # rans = ran_run(rose, dim, min, runs=100000, comps_per_run=50, stats=True)
    # best = visualize2(ei_runs, min)
    # print("ONE WAY ANOVA:", f_oneway(rans, best))

    # plt.savefig("rosen.png", dpi=200)
    # plt.show()


    # Hartmann6D RF
    # dim = hartdim(6)
    # for i in range(100):
    #     auto_setup(hart6, dim, wfname='hart{}'.format(i), opt_label='ei{}'.format(i), host='localhost', acq='ei', name='hart', port=27017, n_bootstraps=1000, predictor="RandomForestRegressor", n_search_points=10000)
    #
    # ranx, rany = ran_run(hart6, dim, min, runs=10000, comps_per_run=50)
    # pd.DataFrame({'x': ranx, 'y': rany}).to_csv("ran_hart.csv")
    # lpad = LaunchPad(host='localhost', port=27017, name='hart')
    # df = pd.DataFrame.from_csv("ran_hart.csv")
    # ranx = df['x']
    # rany = df['y']
    # ei_runs = [getattr(lpad.db, "ei{}".format(i)) for i in range(100)]
    # bm, bs = visualize([ei_runs], min, labels=['EI'], colors=['blue'], limit=50)
    # plt.plot(ranx, rany, color='black', label="Random")
    # plt.tight_layout()
    # plt.legend()
    # print "BEST RANDOM", min(rany)
    # print "BEST OPT", bm, "+-", bs
    # plt.savefig("hart.png", dpi=200)
    # plt.show()

    # rans = ran_run(hart6, dim, min, runs=100000, comps_per_run=50, stats=True)
    # best = visualize2(ei_runs, min)
    # print("ONE WAY ANOVA:", f_oneway(rans, best))


    # Efficiencies graph
    # branin = {'mean': 0.730884202932, 'random': 1.43637817968, 'std': 0.757805941341}
    # rose = {'mean': 16.9325809357, 'std': 64.9309866321, 'random': 28.2279215643}
    # hart = {'mean': -2.43870765313, 'std': 0.460186250753, 'random': -1.75311127331}
    # fig, ax = plt.subplots()
    # r1 = ax.bar(np.arange(3), [branin['random']/branin['mean'], rose['random']/rose['mean'], hart['mean']/hart['random']], 0.5, color='blue')
    # ax.set_xticks(np.arange(3))
    # ax.set_xticklabels(('Branin', 'Rosenbrock', 'Hartmann6'))
    # ax.set_ylabel("Efficiency")



    # RASTRIGIN 50D
    # dim = rastdim(50)
    # for i in range(100):
    #     auto_setup(rast, dim, wfname='rast{}'.format(i), opt_label='ei{}'.format(i), host='localhost', acq='ei', name='rast', port=27017, n_bootstraps=1000, predictor="RandomForestRegressor", n_search_points=10000)
    #
    # ranx, rany = ran_run(rast, dim, min, runs=10000, comps_per_run=50)
    # pd.DataFrame({'x': ranx, 'y': rany}).to_csv("ran_rast.csv")
    # lpad = LaunchPad(host='localhost', port=27017, name='rast')
    # df = pd.DataFrame.from_csv("ran_rast.csv")
    # ranx = df['x']
    # rany = df['y']
    # ei_runs = [getattr(lpad.db, "ei{}".format(i)) for i in range(100)]
    # bm, bs = visualize([ei_runs], min, labels=['EI'], colors=['blue'], limit=50)
    # plt.plot(ranx, rany, color='black')
    # print "BEST RANDOM", min(rany)
    # print "BEST OPT", bm, "+-", bs
    # plt.show()

    # SCHAFFER N4 2D
    # dim = [(-100.0, 100.0), (-100.0, 100.0)]
    # for i in range(100):
    #     auto_setup(schaffer, dim, wfname='scha{}'.format(i), opt_label='ei{}'.format(i), host='localhost', acq='ei', name='scha', port=27017, n_bootstraps=1000, predictor="RandomForestRegressor", n_search_points=10000)
    #
    # ranx, rany = ran_run(schaffer, dim, min, runs=10000, comps_per_run=50)
    # pd.DataFrame({'x': ranx, 'y': rany}).to_csv("ran_scha.csv")
    # lpad = LaunchPad(host='localhost', port=27017, name='scha')
    # df = pd.DataFrame.from_csv("ran_scha.csv")
    # ranx = df['x']
    # rany = df['y']
    # ei_runs = [getattr(lpad.db, "ei{}".format(i)) for i in range(100)]
    # bm, bs = visualize([ei_runs], min, labels=['EI'], colors=['blue'], limit=50)
    # plt.plot(ranx, rany, color='black')
    # print "BEST RANDOM", min(rany)
    # print "BEST OPT", bm, "+-", bs
    # plt.show()