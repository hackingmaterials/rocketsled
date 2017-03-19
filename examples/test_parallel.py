"""
An example of running turboworks optimizations in parallel.
"""

import os
from fireworks import Workflow, Firework, LaunchPad
from turboworks.optimize import OptTask
from turboworks.optdb import OptDB
from turboworks.utils import random_guess
from matplotlib import pyplot as plot
from examples.calculate_task import BasicCalculateTask as CalculateTask


# Get the full path of the directory containing this file
path = os.path.dirname(os.path.realpath(__file__))
# You don't need this if your code is inside a package


dims = [(1, 5), (1, 5), (1, 5)]

# a workflow creator function which takes z and returns a workflow based on z
def wf_creator(z):

    spec = {'A':z[0], 'B':z[1], 'C':z[2], '_z':z}
    Z_dim = dims

    firework1 = Firework([CalculateTask(), OptTask(wf_creator =path + 'test_parallel.wf_creator',
                                                   dimensions=Z_dim,
                                                   opt_label="parallel")], spec=spec)
    return Workflow([firework1])


# try a parallel implementation of turboworks
def load_parallel_wfs(n_processes):
    for i in range(n_processes):
        launchpad.add_wf(wf_creator(random_guess(dims)))


if __name__ == "__main__":
    launchpad = LaunchPad()
    opt_db = OptDB(opt_label="parallel")
    opt_db.clean()  # cleans only previous runs of test_parallel


    # uncomment the line below to reset fireworks
    launchpad.reset('', require_password=False)

    n_processes = 2
    n_runs = 5

    load_parallel_wfs(n_processes)

    minima = []
    for i in range(n_runs):
        sh_output = os.system('rlaunch multi ' + str(n_processes) + ' --nlaunches 1')
        print(sh_output)
        minima.append(opt_db.min.value)

    plot.plot(range(len(minima)), minima)
    plot.ylabel('Best Minimum Value')
    plot.xlabel('Iteration')
    plot.show()






