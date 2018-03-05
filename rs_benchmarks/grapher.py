from matplotlib import pyplot as plt
import pickle
import numpy as np


def depickle(file):
    return pickle.load(open(file, 'rb'))


# lines = ['ran', 'gp', 'rf', 'svr']
lines = ['ran', 'gp', 'rf', 'svr']


# f, (axbranin, axrosen) = plt.subplots(2)
f, axrosen = plt.subplots(1)


colormap = {'gp': 'green', 'ran': 'black', 'rf': 'red', 'svr': 'blue', 'ran2': 'grey'}
algmap = {'gp': 'Gaussian Process', 'ran': 'Random', 'rf': 'Random Forest', 'svr': 'Support Vector'}
titlemap = {'branin': 'Branin-Hoo', 'rosen': 'Rosenbrock'}
for line in lines:
    # for BBFUN in ['branin', 'rosen']:
    for BBFUN in ['rosen']:
        try:
            # ax = axbranin if BBFUN=='branin' else axrosen
            ax = axrosen
            tot_data = depickle('{}_{}.pickle'.format(line, BBFUN))
            print np.asarray(tot_data).shape
            mean = np.mean(tot_data, axis=0)
            std = np.mean(tot_data, axis=0)
            x = range(len(mean))
            # plt.errorbar(x, mean, yerr=std, color=colormap[line])
            ax.semilogy(x, mean, color=colormap[line], label=algmap[line])
            ax.set_ylim([0, 2100])
            # ax.set_title(titlemap[BBFUN])
            ax.set_title("Log")
            ax.set_ylabel('Rosenbrock Value')
            ax.set_xlabel('No. Function Evaluations')
            # if line != 'ran':
            #     ax.fill_between(x, mean-std, mean+std, color=colormap[line], alpha=0.2)
            ax.legend()
        except:
            pass

f.subplots_adjust(hspace=.4)
plt.show()