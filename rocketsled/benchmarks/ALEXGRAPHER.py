import time
import warnings
import datetime
import numpy as np
from matplotlib import pyplot as plt
from skopt.benchmarks import branin, hart6
from rocketsled import auto_setup
from rocketsled.utils import Dtypes
import random
from fireworks import LaunchPad
from itertools import product
import pandas as pd
import math

dtypes = Dtypes()

def visualize(csets, opt, labels, colors, fontfamily="serif", limit=0):

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
        plt.plot(i, mean, label=" Avg best {}: {}".format(labels[l], avgbest), color=colors[l])
        plt.fill_between(i, mean - std, mean + std, color=colors[l], alpha=0.2)

    plt.rc('font', family=fontfamily)
    plt.yscale("log")
    plt.xlabel("fx evaluation")
    plt.ylabel("fx value")
    plt.legend()
    return (mean[-1], std[-1])



def ran_run(func, dims, opt, runs, comps_per_run):
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
    meanbest = np.mean(best, axis=0)
    return ([i + 1 for i in range(comps_per_run)], meanbest)



def rast(X):
    return 10*len(X) + sum([(x**2 - 10 * np.cos(2 * math.pi * x)) for x in X])

def rastdim (dim):
    return [(-5.12, 5.12)] * dim

def hartdim(dim):
    return [(0.0, 1.0)] * dim

def rose(x):
    x = np.asarray(x)
    r = np.sum(100.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0, axis=0)
    return r

def rosedim(dim):
    return [(-5.0, 10.0)] * dim


def schaffer(x):
    return 0.5 + ((math.sin(x[0]**2 - x[1]**2))**2 - 0.5)/\
           ((1.0 + .001 * (x[0]**2 + x[1]**2)) ** 2)


if __name__ == "__main__":

    # BRANIN 2D RF
    # dim = [(-5.0, 10.0), (0.0, 15.0)] # branin
    # # for i in range(100):
    # #     auto_setup(branin, dim, wfname='bran{}'.format(i), opt_label='ei{}'.format(i), host='localhost', acq='ei', name='bran', port=27017, n_bootstraps=1000, predictor="RandomForestRegressor", n_search_points=10000)
    #
    # # ranx, rany = ran_run(branin, dim, min, runs=10000, comps_per_run=50)
    # pd.DataFrame({'x': ranx, 'y': rany}).to_csv("ran_bran.csv")
    lpad = LaunchPad(host='localhost', port=27017, name='bran')
    df = pd.DataFrame.from_csv("ran_bran.csv")
    ranx = df['x']
    rany = df['y']
    hi_runs = [getattr(lpad.db, "ei{}".format(i)) for i in range(100)]
    bm, bs = visualize([hi_runs], min, labels=['EI'], colors=['blue'], limit=50)
    plt.plot(ranx, rany, color='black')
    print "BEST RANDOM", min(rany)
    print "BEST OPT", bm, "+-", bs
    plt.show()


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
    # plt.plot(ranx, rany, color='black')
    # print "BEST RANDOM", min(rany)
    # print "BEST OPT", bm, "+-", bs
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
    # plt.plot(ranx, rany, color='black')
    # print "BEST RANDOM", min(rany)
    # print "BEST OPT", bm, "+-", bs
    # plt.show()



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