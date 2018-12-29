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

from rocketsled import OptTask, MissionControl
from rocketsled.examples.tasks import SumTask

launchpad = LaunchPad(name='rsled')
opt_label = "opt_default"
db_info = {"launchpad": launchpad, "opt_label": opt_label}
x_dim = [(1, 5), (1, 5), (1, 5)]


# a workflow creator function which takes x and returns a workflow based on x
def wf_creator(x):
    spec = {'_x': x}
    # SumTask writes _y field to the spec internally.
    firework1 = Firework([SumTask(), OptTask(**db_info)], spec=spec)
    return Workflow([firework1])


if __name__ == "__main__":
    mc = MissionControl(**db_info)
    mc.reset(hard=True)
    mc.configure(wf_creator=wf_creator, dimensions=x_dim)
    launchpad.reset(password=None, require_password=False)
    launchpad.add_wf(wf_creator([5, 5, 2]))
    rapidfire(launchpad, nlaunches=10, sleep_time=0)
