import numpy as np
import cPickle as pickle
import scipy
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
policy.set_seed(0)



# How to use the interactive mode
simulator = simulator()

''' 1st step (random sampling) '''
actions = policy.random_search(max_num_probes=1, num_search_each_probe=10, simulator=None)
t  = simulator(actions)
policy.write(actions, t)
combo.search.utility.show_search_results(policy.history, 10)

''' 2nd step (random sampling) '''
actions = policy.random_search(max_num_probes=1, num_search_each_probe=10, simulator=None)
t = simulator(actions)
policy.write(actions, t)
combo.search.utility.show_search_results(policy.history, 10)

''' 3rd step (bayesian optimization) '''
actions = policy.bayes_search(max_num_probes=1, num_search_each_probe=10,
                                                      simulator=None, score='EI', interval=0,  num_rand_basis = 0)
t = simulator(actions)   # experiment
policy.write(actions, t) # record new observations
combo.search.utility.show_search_results(policy.history, 10)  # describe search results

predictor = policy.predictor
training = policy.training

actions = policy.bayes_search(max_num_probes=1, num_search_each_probe=10,
                                                      predictor=predictor, training=training,
                                                      simulator=None, score='EI', interval=0,  num_rand_basis = 0)
t = simulator(actions)   # experiment
policy.write(actions, t) # record new observations
combo.search.utility.show_search_results(policy.history, 10)  # describe search results



''' 4-th step (bayesian optimization) '''
actions = policy.bayes_search(max_num_probes=1, num_search_each_probe=10,
                                                      simulator=None, score='EI', interval=0,  num_rand_basis = 0)
t = simulator(actions)   # experiment
policy.write(actions, t) # record new observations
combo.search.utility.show_search_results(policy.history, 10)  # describe search results

predictor = policy.predictor
training = policy.training

actions = policy.bayes_search(max_num_probes=1, num_search_each_probe=10,
                                                      predictor=predictor, training=training,
                                                      simulator=None, score='EI', interval=0,  num_rand_basis = 0)
t = simulator(actions)   # experiment
policy.write(actions, t) # record new observations
combo.search.utility.show_search_results(policy.history, 10)  # describe search results


with open('predictor.dump', 'w') as f:
    pickle.dump(policy.predictor, f)
policy.training.save('training.npz')
policy.history.save('history.npz')

''' delete policy'''
del policy

policy = combo.search.discrete.policy(test_X=X)
policy.load('history.npz', 'training.npz', 'predictor.dump')

''' 5-th probe (bayesian optimization) '''
actions = policy.bayes_search(max_num_probes=1, num_search_each_probe=10, predictor=predictor,
                                                      simulator=None, score='EI', interval=0,  num_rand_basis = 0)
t = simulator(actions)   # experiment
policy.write(actions, t) # record new observations
combo.search.utility.show_search_results(policy.history, 10)  # describe search result