import numpy as np
import combo
from turboworks.discrete_spacify import calculate_discrete_space
import multiprocessing

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug('This is a log message.')

dimensions = [(1,10),(1,10)]
X = calculate_discrete_space(dimensions)

class BBclass:
    def __init__(self):
        self.X = X

    def BBfun(self,x):
        return x[0]*x[1]/3.292884

    def __call__(self, actions):
        self.output = []
        for action in actions:
            self.output.append(self.BBfun(X[action]))
        return np.asarray(self.output)


def get_input_from_actions(actions, X):
    output = []
    for action in actions:
        output.append(X[action])
    return output

def job():
    policy = combo.search.discrete.policy(test_X=np.asarray(X))
    simulator = BBclass()

    ''' RANDOM SAMPLING '''
    actions = policy.random_search(max_num_probes=1)
    t = simulator(actions)
    policy.write(actions, t)
    combo.search.utility.show_search_results(policy.history, 10)

    my_inputs = get_input_from_actions(actions,X)
    print "my_inputs",  my_inputs

    ''' BAYESIAN OPTIMIZATION '''
    actions = policy.bayes_search()
    t = simulator(actions)   # experiment
    policy.write(actions, t) # record new observations
    combo.search.utility.show_search_results(policy.history, 10)  # describe search results

    my_inputs = get_input_from_actions(actions,X)
    print "my_inputs",  my_inputs


if __name__== "__main__":

    jobs=[]
    for i in range(5):
        p = multiprocessing.Process(target=job)
        jobs.append(p)

    for proc in jobs:
        proc.start()

    for proc in jobs:
        proc.join()


    print "done"