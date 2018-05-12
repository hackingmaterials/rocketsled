from __future__ import unicode_literals, division, print_function
"""
Functions for visualizing optimization progress.
"""
import time
import math
import warnings
import datetime
import numpy as np
from matplotlib import pyplot as plt
from rocketsled.utils import Dtypes, latex_float, pareto

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"

def visualize(collection, maximize=False, showbest=True, showmean=True,
              latexify=False, fontfamily="serif", scale='linear', analysis=True,
              print_pareto=False):
    """
    Visualize the progress of an optimization.

    Args:
        collection (pymongo Collection): The pymongo colllection containing your
            rocketsled optimization. For example, if you set you opt_label to
            'opt123', and used the same db as your LaunchPad, you could use
            lpad.db.opt123
        maximize (bool): Whether to plot optimizing for minimum or maximum.
        showbest (bool): Point out the best point on legend and on plot. If more
            than one best point (i.e., multiple equal maxima), show them all. If
            multiobjective, shows best for each objective, and prints the best
            value and x for each objective.
        showmean (bool): Show the mean and standard deviation for the guesses
            as the computations are carried out.
        latexify (bool): Use LaTeX for formatting.
        fontfamily (str): The font family to use for rendering. Choose from
            'serif', 'sans-serif', 'fantasy', 'monospace', or 'cursive'.
        scale (str): Whether to scale the plot's y axis according to log ('log')
            or 'linear' scale.
        analysis (bool): If True, stdouts info from analyze().
        print_pareto (bool): If True, display all Pareto-optimal objective
            values.

    Returns:
        Either None, a matplotlib plot, or a pymongo iterator. See 'mode' for
        details.
    """

    dtypes = Dtypes()
    fxstr = "$f(x)$" if latexify else "f(x)"
    opt = max if maximize else min
    objs = collection.find_one({'index': {'$exists': 1}})['y']
    n_objs = len(objs) if isinstance(objs, (list, tuple)) else 1

    dt = datetime.datetime.now()
    dtdata = [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second]
    timestr = "{}-{}-{} {}:{}.{}".format(*dtdata)
    t0 = time.time()
    if latexify:
        plt.rc('text', usetex=True)
    else:
        plt.rc('text', usetex=False)
    plt.rc('font', family=fontfamily, size=9)


    N_COLS = 3
    # print(int(math.ceil(float(n_objs)/float(N_COLS))))

    if n_objs < N_COLS:
        f, axarr = plt.subplots(n_objs, squeeze=False)
    else:
        f, axarr = plt.subplots(N_COLS, int(math.ceil(n_objs/N_COLS)),
                                squeeze=False)

    docset = collection.find({'index': {'$exists': 1}})
    docs = [None] * docset.count()
    for i, doc in enumerate(docset):
        docs[i] = {'y': doc['y'], 'index': doc['index'], 'x': doc['x']}

    if n_objs > 1:
        Y = np.asarray([doc['y'] for doc in docs])
        pareto_set = Y[pareto(Y, maximize=maximize)].tolist()
        pareto_graph = [(i + 1, doc['y']) for i, doc in enumerate(docs)
                        if doc['y'] in pareto_set]
        pareto_i = [i[0] for i in pareto_graph]

    for obj in range(n_objs):
        ax = axarr[obj % N_COLS, int(math.floor(obj / N_COLS))]

        i = []
        fx = []
        best = []
        mean = []
        std = []
        n = collection.find().count() - 2

        for doc in docs:
            fx.append(doc['y'] if n_objs == 1 else doc['y'][obj])
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

        ax.scatter(i, fx, color='blue', label=fxstr, s=10)
        ax.plot(i, best, color='orange', label="best {} value found so far"
                                                "".format(fxstr))
        if showmean:
            ax.plot(i, mean, color='grey', label = "mean {} value (with std "
                                                    "dev.)".format(fxstr))
            ax.fill_between(i, mean + std, mean - std, color='grey', alpha=0.3)

        ax.set_xlabel("{} evaluation".format(fxstr))
        ax.set_ylabel("{} value".format(fxstr))
        best_val = opt(best)

        if showbest:
            if latexify:
                best_label = "Best value: $f(x) = {}$" \
                             "".format(latex_float(best_val))
            else:
                best_label = "Best value: f(x) = {:.2E}".format(best_val)
            best = collection.find({'y': best_val})
            for b in best:
                bl = None if n_objs > 1 else best_label
                ax.scatter([b['index']], [best_val], color='darkgreen', s=50,
                            linewidth=3, label=bl, facecolors='none',
                            edgecolors='darkgreen')

                artext = "$x = $ [" if latexify else "x = ["
                for i, xi in enumerate(b['x']):
                    if i > 0:
                        artext += ". \mbox{~~~~~}" if latexify else "     "
                    if type(xi) in dtypes.floats:
                        if latexify:
                            artext += "${}$,\n".format(latex_float(xi))
                        else:
                            artext += "{:.2E},\n".format(xi)
                    else:
                        artext += str(xi) + ",\n"

                artext = artext[:-2] + "]"
                objstr = "objective {}".format(obj) if n_objs > 1 else ""
                if maximize:
                    print("max(f(x)) {} is {} at x = {}".format(objstr,
                                                                best_val,
                                                                b['x']))
                else:
                    print("min(f(x)) {} is {} at x = {}".format(objstr,
                                                                best_val,
                                                                b['x']))
                ax.annotate(artext,
                             xy=(b['index'] + 0.5, best_val),
                             xytext=(b['index'] + float(n)/12.0, best_val),
                             arrowprops=dict(color='green'),
                             color='darkgreen',
                             bbox=dict(facecolor='white', alpha=1.0))
        else:
            best_label = ""

        if n_objs > 1:
            pareto_fx = [i[1][obj] for i in pareto_graph]
            ax.scatter(pareto_i, pareto_fx, color='red', label="Pareto optimal",
                       s=20)

        if n_objs > 1:
            ax.set_title("Objective {}: {}".format(obj + 1, best_label))
        ax.set_yscale(scale)

    plt.gcf().set_size_inches(10, 10)
    if analysis:
        print(analyze(collection))

    if print_pareto and n_objs > 1:
        print("Pareto Frontier: {} points".format(len(pareto_set)))
        for i, p in enumerate(pareto_set):
            print(p)

    if n_objs % N_COLS != 0 and n_objs > N_COLS:
        for i in range(n_objs % N_COLS, N_COLS):
            plt.delaxes(axarr[i, -1])

    plt.legend()
    # plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.suptitle("Rocketsled optimization results for {} - "
                 "{}".format(collection.name, timestr), y=0.99)
    plt.show()

def analyze(collection):
    """
    Returns stats about the optimization collection and checks consistency
    of the collection.

    Args:
        collection (pymongo Collection): The pymongo colllection containing your
            rocketsled optimization. For example, if you set you opt_label to
            'opt123', and used the same db as your LaunchPad, you could use
            lpad.db.opt123

    Returns:
        fmtstr (str): The formatted information from the analysis, to print.
    """
    manager = collection.find_one({'lock': {"$exists": 1}})
    qlen = len(manager['queue'])
    lock = manager['lock']
    predictors = {}
    dim = None
    for doc in collection.find({'index': {'$exists': 1},
                                'y': {'$exists': 1, "$ne": "reserved"}}):
        p = doc['predictor']
        if p in predictors:
            predictors[p] += 1
        else:
            predictors[p] = 1
        d = [type(xi) for xi in doc['x'] + doc['z']]
        if not dim:
            dim = d
        elif dim != d:
            warnings.warn("It appears the optimization contained in {} is"
                             "broken, as the x + z dims do not match between"
                             "doc index ({}) and index 0 ({}). To fix, remove"
                             "documents of dissimilar x or z length/type. and "
                             "ensure only one optimization is used for this "
                             "collection!"
                             "".format(collection, d, dim))
    dimdoc = collection.find_one({'index': {'$exists': 1},
                                'y': {'$exists': 1, "$ne": "reserved"}})
    xdim = [type(d) for d in dimdoc['x']]
    zdim = [type(d) for d in dimdoc['z']]
    n_opts = sum(predictors.values())
    n_reserved = collection.find({'y': 'reserved'}).count()
    breakdown = ""
    for p, v in predictors.items():
        predfrac = float(v)/float(n_opts)
        breakdown += "    * {0:.2f}%: ".format(predfrac * 100.0) + p + "\n"

    if not lock:
        lockstr = "DB not locked by any process (no current optimization)."
    else:
        lockstr = "DB locked by PID {}".format(lock)
    zlearn = "" if not zdim else "Only Z data is being used for learning."
    fmtstr = "\nProblem dimension: \n    * X dimensions ({}): {}\n" \
             "    * Z dimensions ({}): {}\n" \
             "{}\n" \
             "Number of Optimizations: {}\n" \
             "Optimizers used (by percentage of optimizations): \n{}" \
             "Number of reserved guesses: {}\n" \
             "Number of waiting optimizations: {}\n" \
             "{}\n".format(len(xdim), xdim, len(zdim), zdim, zlearn, n_opts,
                           breakdown, n_reserved, qlen, lockstr)
    return fmtstr


if __name__ == "__main__":
    from fireworks import LaunchPad
    lpad = LaunchPad(host='localhost', port=27017, name='rsled')
    visualize(lpad.db.opt_default)