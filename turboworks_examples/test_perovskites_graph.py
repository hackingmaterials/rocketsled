from matplotlib import pyplot
import pickle
import numpy as np
from random import randint


# reference lists:

# chemical rules in iterationwise format
ch = [0, 129, 157, 256, 271, 431, 512, 586, 603, 972, 981, 1100, 1165, 1216, 1239, 1258, 1466, 1715, 2258, 3141, 4009]

# 50 run average of random guessing
ran = [0, 763.46, 1682.28, 2685.18, 3536.56, 4604.58, 5920.56, 6555.42, 7667.04, 8412.74, 9228.02, 10016.54, 10796.92,
           11655.7, 12701.22, 13784.92, 14490.68, 15392.34, 16295.12, 17094.34, 18045.58]

def addtofig(ulm, color, label, length=None):
    if not length:
        length = len(ulm['mean'])

    upper = ulm['upper'][0:length]
    lower = ulm['lower'][0:length]
    mean = ulm['mean'][0:length]
    x = range(len(mean))
    pyplot.plot(x, mean, color=color, linewidth=2.5, label=label)
    pyplot.plot(x, lower, color=color, linewidth=0.5, alpha=0.3)
    pyplot.plot(x, upper, color=color, linewidth=0.5, alpha=0.3)
    pyplot.fill_between(x, lower, upper, color=color, alpha=0.13)

def addtofig_iterationwise(ulm, color, label, length=None, single=False, alphamult=1.0):
    if single:
        if not length:
            length = len(ulm)
        mean = ulm[0:length]
    else:
        if not length:
            length = len(ulm['mean'])
        upper = ulm['upper'][:length]
        lower = ulm['lower'][:length]
        mean = ulm['mean'][:length]

    y = range(len(mean))
    pyplot.plot(mean, y, color=color, marker='o', linewidth=2.5, label=label, alpha=alphamult)

    if not single:
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

    minlen = 21
    for i in I:
        if len(i) < minlen:
            minlen = len(i)
    I = [i[:minlen] for i in I]

    if len(I) == 1:
        mean = I[0]
        std = [0]*len(I[0])
    else:
        mean = np.mean(I, axis=0)
        std = np.std(I, axis=0)

    lower = [mean[i] - std[i] for i in range(len(mean))]
    upper = [mean[i] + std[i] for i in range(len(mean))]
    return {'mean': mean, 'lower': lower, 'upper': upper}

def addtofig_individuals(Y, color, label):
    for k, y in enumerate(Y):
        rfwithz = get_stats_iterationwise([y])
        addtofig_iterationwise(rfwithz['mean'], color, label + ' {}'.format(k+1), single=True,
                               alphamult=float(k)/10+0.2)


def depickle(file):
    return pickle.load(open(file, 'rb'))

if __name__=="__main__":
    # rfwithz_Y = depickle('perovskites_RandomForestRegressor_withz_2000iters_5runs.p')
    # rfnoz_Y = depickle('perovskites_RandomForestRegressor_noz_2000iters_5runs.p')

    rfnoz_Y = []
    # for i in range(6):
    #     rfnoz_Y.append(depickle('perovskites_RandomForestRegressor_noz_20cands_20runs.p_{}'.format(i)))
    # rfnoz_stats = get_stats_iterationwise(rfnoz_Y)
    # rfwithz_stats = get_stats_iterationwise(rfwithz_Y)

    # addtofig_iterationwise(rfnoz_stats, 'slategrey', 'RF without z')
    # addtofig_iterationwise(rfwithz_stats, 'dodgerblue', 'RF with z')
    addtofig_iterationwise(ch, 'orange', 'Chemical Rules', single=True)
    addtofig_iterationwise(ran, 'red', 'Random Search', single=True)

    pyplot.xlim(0, 4000)

    pyplot.legend(loc='upper right', prop={'size':8})
    pyplot.xlabel("Iteration")
    pyplot.ylabel("Light Splitter Candidates Found")
    pyplot.title("Comparison of Predictors for Finding Light Splitting Perovskites")
    # pyplot.savefig("{}_withoutz_max.png".format(predictor))
    pyplot.show()