"""
An example of the most basic rocketsled implementation.
This file creates and executes a workflow containing one Firework.

The Firework contains 2 Tasks.
    1. CalculateTask - a task that reads x from the spec and calculates the sum
     of the vector.
    2. OptTask - a task that stores optimiztion data in the db and optimizes
    the next guess.


--------------------------------------------------------------------------
The following workflow is only one Firework (one job), for example purposes.
However, FireWorks and rocketsled are capable of handling more complex
workflows including multiple jobs and advanced dependencies. Please see the
Fireworks and rocketsled documentation pages for more information:

https://hackingmaterials.github.io/rocketsled/
https://materialsproject.github.io/fireworks/
"""

from fireworks.core.rocket_launcher import rapidfire
from fireworks import Workflow, Firework, LaunchPad

from rocketsled import OptTask
from rocketsled.examples.tasks import SumTask
from rocketsled.db import RailsConfig

LPAD = LaunchPad(name='rsled')
X_dim = [(1, 5), (1, 5), (1, 5)]


# a workflow creator function which takes x and returns a workflow based on x
def wf_creator(x):
    spec = {'_x': x}
    # SumTask writes _y field to the spec internally.
    firework1 = Firework([SumTask(), OptTask(launchpad=LPAD)], spec=spec)
    return Workflow([firework1])


if __name__ == "__main__":
    r = RailsConfig(wf_creator='rocketsled.examples.basic.wf_creator',
                    dimensions=X_dim,
                    launchpad=LPAD)
    r.configure()
    LPAD.reset(password=None, require_password=False)
    LPAD.add_wf(wf_creator([5, 5, 2]))
    rapidfire(LPAD, nlaunches=10, sleep_time=0)
