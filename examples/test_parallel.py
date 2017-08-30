"""
An example of running turboworks optimizations in parallel.
"""

import os, random
from fireworks import Workflow, Firework, LaunchPad
from turboworks.optimize import OptTask, random_guess
from examples.calculate_task import BasicCalculateTask as CalculateTask


dims = [(1, 5), (1, 5), (1, 5)]

# a workflow creator function which takes z and returns a workflow based on x
def wf_creator(x):

    spec = {'A': x[0], 'B': x[1], 'C': x[2], '_x_opt': x, '_add_launchpad_and_fw_id': True}
    Z_dim = dims

    firework1 = Firework([CalculateTask(), OptTask(wf_creator='examples.test_parallel.wf_creator',
                                                   dimensions=Z_dim,
                                                   host='localhost',
                                                   port=27017,
                                                   name='turboworks',
                                                   duplicate_check=True,
                                                   opt_label="opt_parallel")], spec=spec)
    return Workflow([firework1])


# try a parallel implementation of turboworks
def load_parallel_wfs(n_processes):
    for i in range(n_processes):
        launchpad.add_wf(wf_creator(random_guess(dims)))


if __name__ == "__main__":

    TESTDB_NAME = 'turboworks'
    launchpad = LaunchPad(name=TESTDB_NAME)
    launchpad.reset(password=None, require_password=False)

    n_processes = 10
    n_runs = 13

    # Should throw an 'Exhausted' error when n_processes*n_runs > 125 (the total space size)

    load_parallel_wfs(n_processes)

    for i in range(n_runs):
        sh_output = os.system('rlaunch -s multi ' + str(n_processes) + ' --nlaunches 1')
        print(sh_output)

    # tear down database
    # launchpad.connection.drop_database(TESTDB_NAME)





