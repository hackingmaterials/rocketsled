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

    firework1 = Firework([CalculateTask(), OptTask(wf_creator='turboworks_examples.test_basic.wf_creator',
                                                   dimensions=X_dim,
                                                   host='localhost',
                                                   port=27017,
                                                   name='opt_default',
                                                   )], spec=spec)
    return Workflow([firework1])


if __name__ == "__main__":

    TESTDB_NAME = 'turboworks'
    launchpad = LaunchPad(name=TESTDB_NAME)
    launchpad.reset(password=None, require_password=False)

    # clean up tw database if necessary
    # todo: should be integrated with launchpad.reset?

    launchpad.add_wf(wf_creator([5, 5, 2]))

    rapidfire(launchpad, nlaunches=10, sleep_time=0)

    # tear down database
    # launchpad.connection.drop_database(TESTDB_NAME)



