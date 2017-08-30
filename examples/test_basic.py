"""
An example of the most basic turboworks implementation.
This file creates and executes a workflow containing one Firework.

The Firework contains 2 Tasks.
    1. CalculateTask - a task that reads A, B, and C from the spec and calculates (A^2 + B^2)/C
    2. OptTask - a task that stores optimiztion data in the db and optimizes the next guess.
"""

from fireworks.core.rocket_launcher import rapidfire
from fireworks import Workflow, Firework, LaunchPad
from turboworks.optimize import OptTask
from calculate_task import BasicCalculateTask as CalculateTask


__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"


# a workflow creator function which takes x and returns a workflow based on x
def wf_creator(x):

    spec = {'A':x[0], 'B':x[1], 'C':x[2], '_x_opt':x}
    X_dim = [(1, 5), (1, 5), (1, 5)]

    # CalculateTask writes _y_opt field to the spec internally.

    firework1 = Firework([CalculateTask(),
                          OptTask(wf_creator='examples.test_basic.wf_creator',
                                  dimensions=X_dim,
                                  host='localhost',
                                  port=27017,
                                  name='turboworks')],
                          spec=spec)

    return Workflow([firework1])

def run_workflows(test_case=False):
    TESTDB_NAME = 'turboworks'
    launchpad = LaunchPad(name=TESTDB_NAME)

    # clean up tw database if necessary
    if test_case:
        getattr(launchpad.connection, TESTDB_NAME).opt_default.drop()
    launchpad.reset(password=None, require_password=False)

    launchpad.add_wf(wf_creator([5, 5, 2]))
    rapidfire(launchpad, nlaunches=10, sleep_time=0)

    # tear down database
    # launchpad.connection.drop_database(TESTDB_NAME)

if __name__ == "__main__":
    run_workflows()



