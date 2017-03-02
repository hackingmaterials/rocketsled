import math

import matplotlib.pyplot as plt
import numpy as np

from old.gp_opt import gp_minimize
from turboworks.dummy import dummy_minimize


def branin(x):
    x1 = x[0]
    x2 = x[1]
    pi = 3.14159
    a = 1
    b = 5.1/ (4 * (pi ** 2))
    c = 5 / pi
    r = 6
    s = 10
    t = 1 / (8 * pi)
    f = a * ((x2 - b * (x1 ** 2) + c * x1 - r) ** 2) + s * (1 - t) * math.cos(x1) + s
    return f

tolerance = 1.05
target = 0.397887
dimensions = [(-5.0,10.0),(0.0,15.0)]

def iteration_bar_graph(runs):
    gp_iterations = []
    dummy_iterations = []
    for run in range(runs):
        dummy_y = []
        gp_y = []
        gp_x = [dummy_minimize(dimensions)]
        dummy_x = [dummy_minimize(dimensions)]

        iterations = 0
        print "Computing gp run {} of {}:".format(run+1, runs)
        while gp_y==[] or gp_y[-1] > tolerance*target:
            gp_y.append(branin(gp_x[-1]))
            gp_x.append(gp_minimize(gp_x,gp_y,dimensions))
            iterations+=1
            # print "GP branin score:", gp_y[-1]
        gp_iterations.append(iterations)

        iterations = 0

        print "Computing dummy run {} of {}:".format(run + 1, runs)
        while dummy_y==[] or dummy_y[-1]>tolerance*target:
            dummy_y.append(branin(dummy_x[-1]))
            dummy_x.append(dummy_minimize(dimensions))
            # print "Dummy brainin score:", dummy_y[-1]
            iterations+=1
        dummy_iterations.append(iterations)

    dummy_mean = np.mean(dummy_iterations)
    dummy_std = np.std(dummy_iterations)
    gp_mean = np.mean(gp_iterations)
    gp_std = np.std(gp_iterations)

    N=1
    ind = np.arange(N)  # the x locations for the groups
    width = 0.4       # the width of the bars
    # rectsgp = plt.bar(ind, gp_mean, width, color='g', yerr=gp_std)
    # rectsdum = plt.bar(ind + width, dummy_mean, width, color='r', yerr=dummy_std)
    x = [0, 0.6]
    y = (gp_mean, dummy_mean)
    e = (gp_std, dummy_std)
    barlist = plt.bar(x,y,width,yerr=e, alpha=0.5,error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2))
    barlist[0].set_color('g')
    barlist[1].set_color('r')
    plt.margins(0.05)
    plt.plot(list(range(2)), [0,0], color='black')


    print "Dummy:", dummy_mean, "+-", dummy_std
    print "GP mean:", gp_mean, "+-", gp_std
    plt.show()

def testbar():
    y = [3, 10, 7, 5, 3, 4.5, 6, 8.1]
    N = len(y)
    x = range(N)
    width = 1 / 1.5
    plt.bar(x, y, width, color="blue")
    plt.show()

def learning_trends(runs, iterations):
    dsl = []
    gsl = []

    for run in range(runs):
        dummy_y = []
        gp_y = []
        gp_x = [dummy_minimize(dimensions)]
        dummy_x = [dummy_minimize(dimensions)]

        print "computing run {} of {}:".format(run+1, runs)

        for iteration in range(iterations):

            gp_y.append(branin(gp_x[-1]))
            gp_x.append(gp_minimize(gp_x, gp_y, dimensions))

            dummy_y.append(branin(dummy_x[-1]))
            dummy_x.append(dummy_minimize(dimensions))

        dsl.append(dummy_y)
        gsl.append(gp_y)

    print "plotting"

    # from pprint import pprint
    # print("DSL")
    # pprint(dsl)
    # print ("GSL")
    # pprint(gsl)

    for i in range(runs):
        plt.plot(list(range(iterations)), dsl[i],'ro', alpha=0.2)
        plt.plot(list(range(iterations)),gsl[i],'go', alpha=0.2)
    plt.ylim([0,20])
    plt.margins(0.01)
    plt.show()


if __name__ == '__main__':
    iteration_bar_graph(50)
    # testbar()
    # learning_trends(40,100)
