import random

import numpy as np
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import Firework, Workflow, FWAction, FireTaskBase
from fireworks.core.rocket_launcher import rapidfire
from fireworks.core.launchpad import LaunchPad

from rocketsled.control import MissionControl
from rocketsled.task import OptTask
from rocketsled.utils import split_xz

# Setting up the FireWorks LaunchPad
launchpad = LaunchPad(name="rsled")
opt_label = "opt_default"
db_info = {"launchpad": launchpad, "opt_label": opt_label}
x_dim = [(-5.0, 5.0), (-5.0, 5.0)]

batch_size = 6


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
            guess, or for batches, a list of best next x guesses.
    """

    # Here is an example custom predictor which is just random
    best_xz_batch = random.sample(XZ_unexplored, k=batch_size)

    # If your guesses return separate descriptors (z), separate them
    best_x_batch = [split_xz(xz, x_dims, x_only=True) for xz in best_xz_batch]

    # Custom predictor should return a list or tuple of choices if possible
    # i.e., all native python types if possible
    # Return only a list of best X guesses (or single X guess for non-batch)
    return best_x_batch


def wf_creator_rosenbrock(x):
    spec = {"_x": x}
    # ObjectiveFuncTask writes _y field to the spec internally.
    firework1 = Firework([RosenbrockTask(), OptTask(**db_info)], spec=spec)
    return Workflow([firework1])


def execute(n_evaluation, predictor_Selected, acquisition_function):
    mc = MissionControl(**db_info)
    launchpad.reset(password=date_, require_password=True)
    mc.reset(hard=True)
    mc.configure(
        wf_creator=wf_creator_rosenbrock,
        dimensions=x_dim,
        acq=acquisition_function,
        predictor=predictor_Selected,
        batch_size=batch_size,
        predictor_kwargs={"batch_size": batch_size}
    )

    for bs in range(batch_size):
        launchpad.add_wf(wf_creator_rosenbrock(
            [np.random.uniform(-5, 5), np.random.uniform(-5, 5)]
        ))

    rapidfire(launchpad, nlaunches=n_evaluation, sleep_time=0)
    plt = mc.plot()
    plt.show()


if __name__ == "__main__":
    date_ = '2021-11-29'

    predictor_Selected = custom_batch_predictor
    acquisition_function = 'lcb'
    n_evaluation = 20
    execute(n_evaluation, predictor_Selected, acquisition_function)
