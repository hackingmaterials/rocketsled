"""
Running a batch optimization with a custom predictor.


Optimizing the 2D Rosenbrock function, which is a 2D
function with one objective to be minimized. There
are no Z descriptors so we use only the X coordinates
for learning.


We show two examples here:

1. Running a batch optimization with a builtin predictor.
2. Using your own custom predictor while still using
    batch optimization.

Change the USE_CUSTOM_PREDICTOR variable False
to use the builtin predictor.


See the documentation for more information on batch
optimization and how it runs.
"""

import random

import numpy as np
from fireworks.core.firework import FireTaskBase, Firework, FWAction, Workflow
from fireworks.core.launchpad import LaunchPad
from fireworks.core.rocket_launcher import rapidfire
from fireworks.utilities.fw_utilities import explicit_serialize

from rocketsled.control import MissionControl
from rocketsled.task import OptTask
from rocketsled.utils import split_xz

# Choose whether to use the custom_batch_predictor
# function or the inbuilt GaussianProcessRegressor
USE_CUSTOM_PREDICTOR = False


# Setting up the FireWorks LaunchPad
launchpad = LaunchPad(name="rsled")
opt_label = "opt_default"
db_info = {"launchpad": launchpad, "opt_label": opt_label}
x_dim = [(-5.0, 5.0), (-5.0, 5.0)]
batch_size = 5


@explicit_serialize
class RosenbrockTask(FireTaskBase):
    _fw_name = "RosenbrockTask"

    def run_task(self, fw_spec):
        x = fw_spec["_x"]
        y = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        return FWAction(update_spec={"_y": y})


def custom_batch_predictor(XZ_explored, Y, x_dims, XZ_unexplored, batch_size=1):
    """
    Returns a prediction for the next best guess. The returned guess will
    be used to construct a new workflow with the workflow creator function.

    The argument names need not be the same shown here, although their
    position must remain the same.

    This particular implementation just returns a series of random
    guesses in the unexplored space.

    Args:
        XZ_explored ([list]): A list of lists; 2D array of samples (rows)
            by features (columns) of points already evaluated in the search
            space. This is training data.
        Y (list): A vector of samples; this is the training output.
        x_dims (list): The dimensions of the search space
        XZ_unexplored([list[): A list of lists; 2D array of samples (rows)
            by features (columns) of points to be predicted. This is the 'test'
            or prediction dataset.

    Returns:
        x (list): A vector representing the set of parameters for the next best
            guess, or for batches, a list of best next x guesses. Number of
            guesses must match batch_size.
    """

    # Here is an example custom predictor which is just random
    best_xz_batch = random.sample(XZ_unexplored, k=batch_size)

    # If your guesses return separate descriptors (z), separate them
    best_x_batch = [split_xz(xz, x_dims, x_only=True) for xz in best_xz_batch]

    # Custom predictor should return a list or tuple of choices if possible
    # i.e., all native python types if possible
    # Return only a list of batch_size best X guesses (or single X guess
    # for non-batch)
    # For example, if batch_size is 5, return 5 best guesses.
    return best_x_batch


def wf_creator_rosenbrock(x):
    spec = {"_x": x}
    # ObjectiveFuncTask writes _y field to the spec internally.
    firework1 = Firework([RosenbrockTask(), OptTask(**db_info)], spec=spec)
    return Workflow([firework1])


if __name__ == "__main__":
    mc = MissionControl(**db_info)
    launchpad.reset(password="2021-11-29", require_password=True)
    mc.reset(hard=True)

    if USE_CUSTOM_PREDICTOR:
        # 1. Running a batch optimization with a builtin predictor.
        mc.configure(
            wf_creator=wf_creator_rosenbrock,
            dimensions=x_dim,
            predictor=custom_batch_predictor,
            batch_size=batch_size,
            predictor_kwargs={"batch_size": batch_size},
        )
    else:
        # 2. Using a builtin predictor
        mc.configure(
            wf_creator=wf_creator_rosenbrock,
            dimensions=x_dim,
            predictor="GaussianProcessRegressor",
            batch_size=batch_size,
        )

    # A batch will only run once rocketsled has seen at
    # least batch_size samples. Every batch_size new
    # evaluations will lead to another batch optimization.
    for bs in range(batch_size):
        launchpad.add_wf(
            wf_creator_rosenbrock(
                [np.random.uniform(-5, 5), np.random.uniform(-5, 5)]
            )
        )

    rapidfire(launchpad, nlaunches=30, sleep_time=0)

    plt = mc.plot()
    plt.show()
