from __future__ import unicode_literals

"""
Functions for visualizing optimization progress.
"""
import numpy as np
from matplotlib import pyplot as plt

def visualize(lpad, opt_label, opt=min, showbest=True, latexify=False):

    fxstr = "$f(x)$" if latexify else "f(x)"

    c = lpad.db[opt_label]
    n = c.count() - 2
    i = np.zeros(n)
    fx = np.zeros(n)
    best = np.zeros(n)
    mean = np.zeros(n)
    std = np.zeros(n)

    ix = 0
    best_val = None
    best_i = None
    for doc in c.find({'index': {'$exists': 1}}):
        fx[ix] = doc['y']
        i[ix] = doc['index']
        best[ix] = opt(fx)

        if best_val != best[ix]:
            best_val = best[ix]
            best_i = i[ix]

        mean[ix] = np.mean(fx[:ix + 1])
        std[ix] = np.std(fx[:ix + 1])
        ix += 1

    plt.title("Rocketsled optimization results")
    plt.scatter(i, fx, color='blue', label=fxstr)
    plt.plot(i, best, color='orange', label="best {} value".format(fxstr))
    plt.plot(i, mean, color='grey', label = "mean {} value (with std "
                                            "dev.)".format(fxstr))
    plt.fill_between(i, mean + std, mean - std, color='grey', alpha=0.3)
    plt.xlabel("Number of {} evaluations".format(fxstr))
    plt.ylabel("{} value".format(fxstr))

    if showbest:
        if latexify:
            best_label = "Best value: $f(x) = {}$".format(best_val)
        else:
            best_label = "Best value: f(x) = {}".format(best_val)
        plt.scatter([best_i], [best_val], color='red', s=70, linewidth=3,
                    label=best_label, facecolors='none', edgecolors='r')

        best = c.find({'y': best_val})
        for b in best:
            if latexify:
                artext = '$x = {}$'.format(b['x'])
            else:
                artext = 'x = {}'.format(b['x'])
            plt.annotate(artext,
                         xy=(b['index'], best_val),
                         xytext=(b['index'] + 2, best_val),
                         arrowprops=dict(facecolor='red', shrink=0.05),
                         color='red')

    plt.legend()
    plt.show()



if __name__ == "__main__":
    from fireworks import LaunchPad
    lpad = LaunchPad(host='localhost', port=27017, name='ROCKETSLED_EXAMPLES')
    visualize(lpad, 'opt_auto', opt=min, latexify=True)