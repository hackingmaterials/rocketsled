from matplotlib import pyplot
import pickle
import numpy as np

def addtofig(ulm, length, color, label):

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

def get_stats(Y):
    mean = np.mean(Y, axis=0)
    std = np.std(Y, axis=0)
    lower = [mean[i] - std[i] for i in range(len(mean))]
    upper = [mean[i] + std[i] for i in range(len(mean))]
    return {'mean': mean, 'lower': lower, 'upper': upper}


def depickle(file):
    return pickle.load(open(file, 'rb'))['Y']

if __name__=="__main__":
    Y_rf_withz = depickle('perovskites_RandomForestRegressor_noz_1000iters_2runs.p')
    rf_withz = get_stats(Y_rf_withz)

    addtofig(rf_withz, None, 'dodgerblue', 'RF with z')


    pyplot.legend(loc='upper right', prop={'size':8})
    pyplot.xlabel("Iteration")
    pyplot.ylabel("Light Splitter Candidates Found")
    pyplot.title("Comparison of Random Forest with and without z in OptTask for Perovskite Light Splitter")
    # pyplot.title("{} Predictor Performance".format(predictor))
    # pyplot.savefig("{}_withoutz_max.png".format(predictor))
    pyplot.show()