from __future__ import unicode_literals

"""
Functions for visualizing optimization progress.
"""
import numpy as np
from matplotlib import pyplot as plt

def visualize(collection, opt=min, showbest=True, latexify=False,
              fontfamily="serif", mode='show'):
    """
    Visualize the progress of an optimization.

    Args:
        collection (pymongo Collection): The pymongo colllection containing your
            rocketsled optimization. For example, if you set you opt_label to
            'opt123', and used the same db as your LaunchPad, you could use
            lpad.db.opt123
        opt (builtin): Whether to plot optimizing for minimum or maximum. Use
            builtin min for minima, and builtin max for maxima.
        showbest (bool): Point out the best point on legend and on plot. If more
            than one best point (i.e., multiple equal maxima), show them all.
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

    i = []
    fx = []
    best = []
    mean = []
    std = []

    best_val = None
    best_i = None
    for doc in collection.find({'index': {'$exists': 1}}):
        fx.append(doc['y'])
        i.append(doc['index'])
        best.append(opt(fx))

        if best_val != best[-1]:
            best_val = best[-1]
            best_i = i[-1]

        mean.append(np.mean(fx))
        std.append(np.std(fx))

    mean = np.asarray(mean)
    std = np.asarray(std)

    if latexify:
        plt.rc('text', usetex=True)
    else:
        plt.rc('text', usetex=False)

    plt.rc('font', family=fontfamily)
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
        plt.scatter([best_i], [best_val], color='green', s=70, linewidth=3,
                    label=best_label, facecolors='none', edgecolors='green')

        best = collection.find({'y': best_val})
        for b in best:
            if latexify:
                artext = '$x = {}$'.format(b['x'])
            else:
                artext = 'x = {}'.format(b['x'])
            plt.annotate(artext,
                         xy=(b['index'] + 0.5, best_val),
                         xytext=(b['index'] + 5, best_val),
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

if __name__ == "__main__":
    from fireworks import LaunchPad
    lpad = LaunchPad(host='localhost', port=27017, name='ROCKETSLED_EXAMPLES')
    visualize(lpad.db.opt_auto, opt=max)