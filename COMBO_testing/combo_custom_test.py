
"""
Testing the ability of combo on a random BB function
"""


import numpy as np
import combo
from turboworks.discrete_spacify import calculate_discrete_space


dimensions = [(1,10),(1,10)]
X = calculate_discrete_space(dimensions)

class BBclass:
    def __init__(self):
        self.X = X

    def BBfun(self,x):
        return x[0]*x[1]/3.292884

    def __call__(self, action):
        return self.BBfun(X[action])

policy = combo.search.discrete.policy(test_X=np.asarray(X))
policy.set_seed(0)
res = policy.random_search(max_num_probes=5, simulator=BBclass())

res = policy.bayes_search(max_num_probes=20, simulator=BBclass(), score='TS',
                          interval=20, num_rand_basis=5000)