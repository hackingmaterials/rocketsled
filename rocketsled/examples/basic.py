"""
An example of the most basic rocketsled implementation.
This file creates and executes a workflow containing one Firework.

The Firework contains 2 Tasks.
    1. ObjectiveFuncTask - a task that reads x from the spec and
        updates y in the spec, according to a simple known function.
    2. OptTask - a task that stores optimiztion data in the db and optimizes
    the next guess.

--------------------------------------------------------------------------
The following workflow is only one Firework (one job), for example purposes.
However, FireWorks and rocketsled are capable of handling more complex
workflows including multiple jobs and advanced dependencies. Please see the
complex example, or the Fireworks and rocketsled documentation pages for more
information:

https://hackingmaterials.github.io/rocketsled/
https://materialsproject.github.io/fireworks/
"""
from fireworks import FireTaskBase, Firework, FWAction, LaunchPad, Workflow
from fireworks.core.rocket_launcher import rapidfire
from fireworks.utilities.fw_utilities import explicit_serialize

from rocketsled import MissionControl, OptTask

# Setting up the FireWorks LaunchPad
launchpad = LaunchPad(name="rsled")
opt_label = "opt_default"
db_info = {"launchpad": launchpad, "opt_label": opt_label}

# We constrain our dimensions to 3 integers, each between 1 and 5
x_dim = [(1, 5), (1, 5), (1, 5)]


@explicit_serialize
class ObjectiveFuncTask(FireTaskBase):
    """
    An example task which just evaluates the following simple function:

    f(x) = x[0] * x[1] / x[2]

    Replace this code with your objective function if your objective function
    is relatively simple (i.e., only needs one Firework).
    """

    _fw_name = "ObjectiveFuncTask"

    def run_task(self, fw_spec):
        x = fw_spec["_x"]
        y = x[0] * x[1] / x[2]
        return FWAction(update_spec={"_y": y})


def wf_creator(x):
    """
    The workflow creator function required by rocketsled.

    This wf_creator takes in an input vector x and returns a workflow which
    calculates y, the output. The requirements for using this wf_creator
    with rocketsled are:

    1. OptTask is passed into a FireWork in the workflow
    2. The fields "_x" and "_y" are written to the spec of the FireWork
        containing OptTask.
    3. You use MissionControl's "configure" method to set up the optimization,
        and pass in wf_creator as it's first argument.

    Args:
        x (list): The wf_creator input vector. In this example, it is just 3
            integers between 1 and 5 (inclusive).

    Returns:
        (Workflow): A workflow containing one FireWork (two FireTasks) which
            is automatically set up to run the optimization loop.

    """
    spec = {"_x": x}
    # ObjectiveFuncTask writes _y field to the spec internally.
    firework1 = Firework([ObjectiveFuncTask(), OptTask(**db_info)], spec=spec)
    return Workflow([firework1])


if __name__ == "__main__":
    # Make a MissionControl object
    mc = MissionControl(**db_info)

    # Reset the launchpad and optimization db for this example
    launchpad.reset(password=None, require_password=False)
    mc.reset(hard=True)

    # Configure the optimization db with MissionControl
    mc.configure(wf_creator=wf_creator, dimensions=x_dim)

    # Run the optimization loop 10 times.
    launchpad.add_wf(wf_creator([5, 5, 2]))
    rapidfire(launchpad, nlaunches=10, sleep_time=0)

    # Examine results
    plt = mc.plot()
    plt.show()
