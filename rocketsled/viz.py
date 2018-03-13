from __future__ import unicode_literals
"""
Functions for visualizing optimization progress.
"""
import time
import warnings
import datetime
import numpy as np
from matplotlib import pyplot as plt

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"

def visualize(collection, maximize=False, showbest=True, showmean=True,
              latexify=False, fontfamily="serif", mode='show'):
    """
    Visualize the progress of an optimization.

    Args:
        collection (pymongo Collection): The pymongo colllection containing your
            rocketsled optimization. For example, if you set you opt_label to
            'opt123', and used the same db as your LaunchPad, you could use
            lpad.db.opt123
        maximize (bool): Whether to plot optimizing for minimum or maximum.
        showbest (bool): Point out the best point on legend and on plot. If more
            than one best point (i.e., multiple equal maxima), show them all.
        showmean (bool): Show the mean and standard deviation for the guesses
            as the computations are carried out.
        latexify (bool): Use LaTeX for formatting.
        fontfamily (str): The font family to use for rendering. Choose from
            'serif', 'sans-serif', 'fantasy', 'monospace', or 'cursive'.
        mode (str): What to do with the plot/data. Set 'show' to show, 'return'
            to return the plot object, or 'best' to return a pymongo iterator
            for the documents of the best scoring function evaluations.

    Returns:
        Either None, a matplotlib plot, or a pymongo iterator. See 'mode' for
        details.
    """
    fxstr = "$f(x)$" if latexify else "f(x)"
    opt = max if maximize else min

    i = []
    fx = []
    best = []
    mean = []
    std = []
    n = collection.find().count() - 2

    dt = datetime.datetime.now()
    dtdata = [dt.hour, dt.minute, dt.second, dt.day, dt.month, dt.year]
    timestr = "{}:{}.{} {}/{}/{}".format(*dtdata)

    t0 = time.time()

    for doc in collection.find({'index': {'$exists': 1}}):
        fx.append(doc['y'])
        i.append(doc['index'])
        best.append(opt(fx))
        mean.append(np.mean(fx))
        std.append(np.std(fx))

        if time.time() - t0 > 60:
            warnings.warn("Gathering data from the db is taking a while. Ensure"
                          "the latency to your db is low and the bandwidth"
                          "is as high as possible!")

    mean = np.asarray(mean)
    std = np.asarray(std)

    if latexify:
        plt.rc('text', usetex=True)
    else:
        plt.rc('text', usetex=False)

    plt.rc('font', family=fontfamily)
    plt.title("Rocketsled optimization results for {} - "
              "{}".format(collection.name, timestr))
    plt.scatter(i, fx, color='blue', label=fxstr)
    plt.plot(i, best, color='orange', label="best {} value".format(fxstr))

    if showmean:
        plt.plot(i, mean, color='grey', label = "mean {} value (with std "
                                                "dev.)".format(fxstr))
        plt.fill_between(i, mean + std, mean - std, color='grey', alpha=0.3)

    plt.xlabel("{} evaluation".format(fxstr))
    plt.ylabel("{} value".format(fxstr))
    best_val = opt(best)

    if showbest:
        if latexify:
            best_label = "Best value: $f(x) = {:.2E}$".format(best_val)
        else:
            best_label = "Best value: f(x) = {:.2E}".format(best_val)

        best = collection.find({'y': best_val})
        for b in best:
            plt.scatter([b['index']], [best_val], color='green', s=70,
                        linewidth=3, label=best_label, facecolors='none',
                        edgecolors='green')
            if latexify:
                artext = '$x = {}$'.format(b['x'])
            else:
                artext = 'x = {}'.format(b['x'])
            if maximize:
                print("argmax(f(x)) is {}".format(artext))
            else:
                print("argmin(f(x)) is {}".format(artext))
            plt.annotate(artext,
                         xy=(b['index'] + 0.5, best_val),
                         xytext=(b['index'] + float(n)/12.0, best_val),
                         arrowprops=dict(color='green'),
                         color='green',
                         bbox=dict(facecolor='white', alpha=1.0))
    plt.legend()
    if mode=='show':
        plt.show()
    elif mode=='return':
        return plt
    elif mode=='best':
        return collection.find({'y': best_val})
    else:
        return ValueError("Please specify the mode as 'show', 'return', or "
                          "'best'.")


if __name__ == "__main__":
    from fireworks import LaunchPad
    lpad = LaunchPad(host='localhost', port=27017, name='acqtest')
    visualize(lpad.db.ei, maximize=False)