import numpy as np
import cPickle as pickle
import scipy
import combo
import os
import urllib
import matplotlib.pyplot as plt
import combo


def download():
    if not os.path.exists('data/s5-210.csv'):

        if not os.path.exists('data'):
            os.mkdir('data')

        print('Downloading...')
        urllib.urlretrieve('http://www.tsudalab.org/files/s5-210.csv', 'data/s5-210.csv')
        print('Done')


def load_data():
    download()
    A =  np.asarray( np.loadtxt('data/s5-210.csv',skiprows=1,delimiter=',') )
    X = A[:,0:3]
    t  = -A[:,3]
    return X, t


# Load the data.
# X is the N x d dimensional matrix. Each row of X denotes the d-dimensional feature vector of search candidate.
# t is the N-dimensional vector that represents the corresponding negative energy of search candidates.
# ( It is of course unknown in practice. )
X, t = load_data()

# Normalize the mean and standard deviation along the each column of X to 0 and 1, respectively
X = combo.misc.centering(X)


# Declare the class for calling the simulator.
# In this tutorial, we simply refer to the value of t.
# If you want to apply combo to other problems, you have to customize this class.
class simulator:
    def __init__(self):
        _, self.t = load_data()

    def __call__(self, action):
        return self.t[action]



# Design of policy

# Declaring the policy by
policy = combo.search.discrete.policy(test_X=X)
# test_X is the set of candidates which is represented by numpy.array.
# Each row vector represents the feature vector of the corresponding candidate

# set the seed parameter
policy.set_seed( 0 )

# If you want to perform the initial random search before starting the Bayesian optimization,
# the random sampling is performed by

res = policy.random_search(max_num_probes=20, simulator=simulator())
# Input:
# max_num_probes: number of random search
# simulator = simulator
# output: combo.search.discreate.results (class)


# single query Bayesian search
# The single query version of COMBO is performed by
res = policy.bayes_search(max_num_probes=80, simulator=simulator(), score='TS',
                                                  interval=20, num_rand_basis=5000)

# Input
# max_num_probes: number of searching by Bayesian optimization
# simulator: the class of simulator which is defined above
# score: the type of aquision funciton. TS, EI and PI are available
# interval: the timing for learning the hyper parameter.
#               In this case, the hyper parameter is learned at each 20 steps
#               If you set the negative value to interval, the hyper parameter learning is not performed
#               If you set zero to interval, the hyper parameter learning is performed only at the first step
# num_rand_basis: the number of basis function. If you choose 0,  ordinary Gaussian process runs