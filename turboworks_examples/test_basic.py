"""
An example of the most basic turboworks implementation.
This file creates and executes a workflow containing one Firework.

The Firework contains 2 Tasks.
    1. CalculateTask - a task that reads A, B, and C from the spec and calculates (A^2 + B^2)/C
    2. OptTask - a task that stores optimiztion data in the db and optimizes the next guess.
"""
import os
from fireworks.core.rocket_launcher import rapidfire
from fireworks import Workflow, Firework, LaunchPad
from turboworks.optimize import OptTask
from calculate_task import BasicCalculateTask as CalculateTask


__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"


# a workflow creator function which takes z and returns a workflow based on z
def wf_creator(z):

    spec = {'A':z[0], 'B':z[1], 'C':z[2], '_z':z}
    Z_dim = [(1, 5), (1, 5), (1, 5)]

    #CalculateTask writes _y field to the spec internally.

    firework1 = Firework([CalculateTask(), OptTask(wf_creator='turboworks_examples.test_basic.wf_creator',
                                                   dimensions=Z_dim)], spec=spec)
    return Workflow([firework1])


if __name__ == "__main__":

    TESTDB_NAME = 'turboworks'
    launchpad = LaunchPad(name=TESTDB_NAME)
    launchpad.reset(password=None, require_password=False)

    # clean up tw database if necessary
    # todo: should be integrated with launchpad.reset?

    launchpad.add_wf(wf_creator([5, 5, 2]))

    rapidfire(launchpad, nlaunches=10, sleep_time=0)



