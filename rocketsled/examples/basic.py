from __future__ import unicode_literals, print_function, unicode_literals

"""
An example of the most basic rocketsled implementation.
This file creates and executes a workflow containing one Firework.

The Firework contains 2 Tasks.
    1. CalculateTask - a task that reads x from the spec and calculates the sum of the vector.
    2. OptTask - a task that stores optimiztion data in the db and optimizes the next guess.
"""

from fireworks.core.rocket_launcher import rapidfire
from fireworks import Workflow, Firework, LaunchPad
from rocketsled import OptTask
from rocketsled.examples.tasks import SumTask


LPAD = LaunchPad(name='rsled')
X_dim = [(1, 5), (1, 5), (1, 5)]

# a workflow creator function which takes x and returns a workflow based on x
def wf_creator(x):

    spec = {'_x':x}

    # SumTask writes _y field to the spec internally.
    firework1 = Firework([SumTask(),
                          OptTask(wf_creator='rocketsled.examples.basic.'
                                             'wf_creator',
                                  dimensions=X_dim,
                                  lpad=LPAD)],
                          spec=spec)

    return Workflow([firework1])

if __name__ == "__main__":
    LPAD.reset(password=None, require_password=False)
    LPAD.add_wf(wf_creator([5, 5, 2]))
    rapidfire(LPAD, nlaunches=10, sleep_time=0)


