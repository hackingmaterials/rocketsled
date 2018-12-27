"""
Examples of using extra rocketsled features.

In this task, the calculation and optimization includes a categorical dimension,
a function to fetch extra features (get_z), a custom predictor function, extra 
arguments to the workflow creator, duplicate checking enabled, and a custom 
storage location for the optimization data. Also, a demo of how to use the lpad 
kwarg to store optimization data based on a Firework's LaunchPad object.

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
from rocketsled.examples.tasks import MixedCalculateTask
import random

lpad = LaunchPad(name='rsled')
opt_label = "opt_extras"
dims = [(1, 2), (1, 2), (1, 2), ("red", "green", "blue")]


# use a wf_creator function with more arguments...
def wf_creator(x, my_arg, my_kwarg=1):
    fw1_spec = {'A': x[0], 'B': x[1], 'C': x[2], 'D': x[3], '_x': x}

    # CalculateTask writes _y field to the spec internally.

    firework1 = Firework([MixedCalculateTask(),
                          OptTask(wf_creator='rocketsled.examples.extras.'
                                             'wf_creator',
                                  dimensions=dims,
                                  get_z='rocketsled.examples.extras.get_z',
                                  predictor='rocketsled.examples.extras.'
                                            'example_predictor',
                                  max=True,
                                  lpad=lpad,
                                  wf_creator_args=[my_arg * 3],
                                  wf_creator_kwargs={'my_kwarg': my_kwarg * 2},
                                  duplicate_check=True,
                                  opt_label=opt_label,
                                  retrain_interval=5)],
                         spec=fw1_spec)
    return Workflow([firework1])


def get_z(x):
    """
    An optional function which returns extra information 'z' from the unique
    vector 'x'.

    Args:
        x (list): The input x vector, which must be found in the search space
            defined by the dimensions of the problem.

    Returns:
        (list): The z-function response. In this case, a dummy response of
            floats is returned; however, the z-function can return floats,
            ints, and categorical variables, or any combination thereof.

    """
    return [x[0] * 2.1, x[2] ** 3.4]


def example_predictor(XZ_explored, Y, x_dims, XZ_unexplored):
    """
    An example predictor function. Replace the code inside with your favorite
    optimization algorithm (this example is just random search!).

    Args:
        XZ_explored ([list]): The list of explored points (including both x and
            the optional z-vector, concatenated).
        Y ([float] or [list]). The objective function response. Each objective
            function dimension response is required if multiobjective.
        x_dims [list, tuple]: The dimensions of the search space of x.
        XZ_unexplored ([list]): The list of unexplored points (including both x
            and the optional z vector, concatenated).
    Returns:
        (list) The optimizer-recommended best guess.

    """
    # custom optimizer code goes here
    return random.choice(XZ_unexplored)


if __name__ == "__main__":
    # clean up tw database if necessary
    lpad.reset(password=None, require_password=False)

    lpad.add_wf(wf_creator([1, 1, 2, "red"], 3, my_kwarg=1))

    # if n_launches > 23 for this particular example, the search space will be
    # exhausted and OptTask will throw an exception
    rapidfire(lpad, nlaunches=23, sleep_time=0)
