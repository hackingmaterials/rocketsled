import os
from fireworks import Workflow, Firework, LaunchPad
from turboworks.optimize import OptTask, random_guess
from examples.calculate_task import BasicCalculateTask as CalculateTask

opt_label = "opt_batch"
batch_size = 5
n_runs = 10
X_dim = [(1, 5), (1, 5), (1, 5)]


def wf_creator(x):
    spec = {'A': x[0], 'B': x[1], 'C': x[2], '_x_opt': x}

    # CalculateTask writes _y_opt field to the spec internally.

    firework1 = Firework([CalculateTask(),
                          OptTask(wf_creator='examples.test_batch.wf_creator',
                                  dimensions=X_dim,
                                  host='localhost',
                                  port=27017,
                                  name='turboworks',
                                  opt_label=opt_label,
                                  batch_size=batch_size)],
                         spec=spec)

    return Workflow([firework1])

def load_workflows(test_case=False):
    TESTDB_NAME = 'turboworks'
    launchpad = LaunchPad(name=TESTDB_NAME)

    # clean up tw database if necessary
    if test_case:
        getattr(getattr(launchpad.connection, TESTDB_NAME), opt_label).drop()
    launchpad.reset(password=None, require_password=False)

    # load 10 batch workflows onto the launchpad
    for _ in range(n_runs):
        launchpad.add_wf(wf_creator(random_guess(X_dim)))

    # To launch them, either run run_workflows() or use a terminal to type "rlaunch -s singleshot'

def run_workflows():
    for i in range(n_runs):
        sh_output = os.system('rlaunch -s singleshot')
        print(sh_output)

if __name__ == "__main__":
    # load_workflows()
    run_workflows()