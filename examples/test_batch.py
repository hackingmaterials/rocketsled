import os
from fireworks import Workflow, Firework, LaunchPad
from fireworks.core.rocket_launcher import rapidfire
from rocketsled.optimize import OptTask, random_guess
from examples.calculate_task import BasicCalculateTask as CalculateTask

opt_label = "opt_batch"
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
                                  batch_size=5)],
                         spec=spec)

    return Workflow([firework1])

def run_workflows(test_case=False):
    TESTDB_NAME = 'turboworks'
    launchpad = LaunchPad(name=TESTDB_NAME)

    # clean up tw database if necessary
    if test_case:
        getattr(getattr(launchpad.connection, TESTDB_NAME), opt_label).drop()
    launchpad.reset(password=None, require_password=False)

    # load 10 batch workflows onto the launchpad
    for _ in range(10):
        launchpad.add_wf(wf_creator(random_guess(X_dim)))

    rapidfire(launchpad, nlaunches=10, sleep_time=0)

if __name__ == "__main__":
    run_workflows()