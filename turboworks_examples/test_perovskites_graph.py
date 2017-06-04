from matplotlib import pyplot
import pickle
import numpy as np
from random import randint


# reference lists:

# chemical rules in iterationwise format
ch = [0, 129, 157, 256, 271, 431, 512, 586, 603, 972, 981, 1100, 1165, 1216, 1239, 1258, 1466, 1715, 2258, 3141, 4009]

def addtofig(ulm, color, label, length=None):

    if not length:
        length = len(ulm['mean']) - 1

    upper = ulm['upper'][0:length]
    lower = ulm['lower'][0:length]
    mean = ulm['mean'][0:length]
    x = range(len(mean))
    pyplot.plot(x, mean, color=color, linewidth=2.5, label=label)
    pyplot.plot(x, lower, color=color, linewidth=0.5, alpha=0.3)
    pyplot.plot(x, upper, color=color, linewidth=0.5, alpha=0.3)
    pyplot.fill_between(x, lower, upper, color=color, alpha=0.13)

def addtofig_iterationwise(ulm, color, label, length=None):
    if not length:
        length = len(ulm['mean']) - 1

    upper = ulm['upper'][0:length]
    lower = ulm['lower'][0:length]
    mean = ulm['mean'][0:length]
    y = range(len(mean))
    pyplot.plot(mean, y, color=color, marker='o', linewidth=2.5, label=label)
    pyplot.plot(lower, y, color=color, linewidth=0.5, alpha=0.3)
    pyplot.plot(upper, y, color=color, linewidth=0.5, alpha=0.3)
    pyplot.fill_betweenx(y, lower, upper, color=color, alpha=0.13)

def get_stats(Y):
    mean = np.mean(Y, axis=0)
    std = np.std(Y, axis=0)
    lower = [mean[i] - std[i] for i in range(len(mean))]
    upper = [mean[i] + std[i] for i in range(len(mean))]
    return {'mean': mean, 'lower': lower, 'upper': upper}

def get_stats_iterationwise(Y):
    # finds stats in the form of the genetic algorithms paper
    maxcands = np.amax(Y)
    I = []
    for y in Y:
        i = [0]
        for j, yi in enumerate(y):
            if j!=0 and yi > y[j-1] and yi < maxcands:
                i.append(j)
        I.append(i)
    mean = np.mean(I, axis=0)
    std = np.std(I, axis=0)
    lower = [mean[i] - std[i] for i in range(len(mean))]
    upper = [mean[i] + std[i] for i in range(len(mean))]
    return {'mean': mean, 'lower': lower, 'upper': upper}

def depickle(file):
    return pickle.load(open(file, 'rb'))

if __name__=="__main__":
    Y_rf_withz = depickle('perovskites_RandomForestRegressor_withz_5000iters_20runs.p')
    Y_rf_noz = depickle('perovskites_RandomForestRegressor_noz_5000iters_20runs.p')
    Y_ran = depickle('perovskites_random_guess_noz_5000iters_20runs.p')
    rf_withz = get_stats_iterationwise(Y_rf_withz)
    addtofig_iterationwise(rf_withz, 'dodgerblue', 'RF with z')

    #todo: run with train/test points maxed out

    pyplot.legend(loc='upper right', prop={'size':8})
    pyplot.xlabel("Iteration")
    pyplot.ylabel("Light Splitter Candidates Found")
    pyplot.title("Comparison of Random Forest with and without z in OptTask for Perovskite Light Splitter")
    # pyplot.title("{} Predictor Performance".format(predictor))
    # pyplot.savefig("{}_withoutz_max.png".format(predictor))
    pyplot.show()