from matplotlib import pyplot
import pickle
import numpy as np
from random import randint
import numpy
import matplotlib


# reference lists:

# chemical rules in iterationwise format
ch = [0, 129, 157, 256, 271, 431, 512, 586, 603, 972, 981, 1100, 1165, 1216, 1239, 1258, 1466, 1715, 2258, 3141, 4009]

# 50 run average of random guessing
ran = [0, 763.46, 1682.28, 2685.18, 3536.56, 4604.58, 5920.56, 6555.42, 7667.04, 8412.74, 9228.02, 10016.54, 10796.92,
           11655.7, 12701.22, 13784.92, 14490.68, 15392.34, 16295.12, 17094.34, 18045.58]
ran2 = [901.4*i for i in range(21)]

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

def addtofig_iterationwise(ulm, color, label, length=None, single=False, alphamult=1.0, markersize=4.0, **kwargs):
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
    pyplot.plot(mean, y, color=color, marker='o', linewidth=2.0, label=label, alpha=alphamult, markersize=markersize, **kwargs)

    if not single:
        pyplot.plot(lower, y, color=color, linewidth=0.5, alpha=0.3)
        pyplot.plot(upper, y, color=color, linewidth=0.5, alpha=0.3)
        pyplot.fill_betweenx(y, lower, upper, color=color, alpha=0.13)

def get_stats(Y):
    minlen = 18920
    for y in Y:
        if len(y) < minlen:
            minlen = len(y)

    Y = [y[:minlen] for y in Y]

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

        # if the very first guess was a good candidate
        if y[0] != 0:
            y = [0] + y

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

def addtofig_individuals(Y, label, color=None):
    for k, y in enumerate(Y):
        rfwithz = get_stats_iterationwise([y])
        if color:
            addtofig_iterationwise(rfwithz['mean'], color, label + ' {}'.format(k+1), single=True,
                                   alphamult=float(k)/10+0.12)
        else:
            addtofig_iterationwise(rfwithz['mean'], color=numpy.random.rand(3,),
                                   label=label + ' {}'.format(k + 1), single=True)


def depickle(file):
    return pickle.load(open(file, 'rb'))

if __name__=="__main__":
    font = {'family': 'Helvetica',
            'weight': 'medium',
            'size': 11}

    matplotlib.rc('font', **font)

    rf_noex_noz = depickle('RFR_noex_noz.p')
    rf_noex_withz = depickle('RFR_noex_withz.p')
    rf_withex_noz = depickle('RFR_withex_noz.p')
    rf_withex_withz = depickle('RFR_withex_withz.p')

    rf_noex_noz_stats = get_stats_iterationwise(rf_noex_noz)
    rf_noex_withz_stats = get_stats_iterationwise(rf_noex_withz)
    rf_withex_noz_stats = get_stats_iterationwise(rf_withex_noz)
    rf_withex_withz_stats = get_stats_iterationwise(rf_withex_withz)


    print "Efficiencies to 20"
    print "No ex, no z:", 18027/ rf_noex_noz_stats['mean'][-1]
    print "No ex, wi z:",18027/ rf_noex_withz_stats['mean'][-1]
    print "Wi ex, no z:",18027/ rf_withex_noz_stats['mean'][-1]
    print "Wj ex, wi z:",18027/ rf_withex_withz_stats['mean'][-1]
    print "Chem rules", 18027.0 / ch[-1]

    print "Efficiencies to 10"
    print "No ex, no z:", 18027 / rf_noex_noz_stats['mean'][10] / 2
    print "No ex, wi z:", 18027 / rf_noex_withz_stats['mean'][10]  / 2
    print "Wi ex, no z:", 18027 / rf_withex_noz_stats['mean'][10]  / 2
    print "Wj ex, wi z:", 18027 / rf_withex_withz_stats['mean'][10]  / 2
    print "Chem rules", 18027.0 / ch[10]  / 2


    addtofig_iterationwise(rf_noex_noz_stats, 'green', 'RF without chem rules without z')
    addtofig_iterationwise(rf_noex_withz_stats, 'red', 'RF without chem rules with z')
    addtofig_iterationwise(rf_withex_noz_stats, 'blue', 'RF with chem rules without z')
    addtofig_iterationwise(rf_withex_withz_stats, 'orange', 'RF with chem rules with z')
    addtofig_iterationwise(ch, 'black', 'Chemical Rules', single=True)
    addtofig_iterationwise(ran2, 'grey', 'Random Search', single=True, markersize=0.1)

    pyplot.xlim(0, 4200)

    pyplot.yticks(range(21))
    # pyplot.legend(loc='upper center', prop={'size':font['size']})
    pyplot.legend(loc='lower right', prop={'size':font['size']})

    # pyplot.xlabel("Calculations")
    pyplot.xlabel('Expensive Function Calculations')
    pyplot.ylabel("Solar Water Splitters Found")
    # pyplot.title("Comparison of Predictors for Finding Solar Water Splitting Perovskites")
    # pyplot.savefig("perovskites.png", dpi=400)
    pyplot.show()