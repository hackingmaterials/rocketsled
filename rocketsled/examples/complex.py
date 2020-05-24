"""
Running a rocketsled optimization with a multi-part workflow, multi-objective
objective function, z-features, as well as more advanced configuration.

Our workflow to optimize now has two Fireworks, each with one FireTask. The
first firework runs the 'expensive' objective function, and the second firework
runs only the optimization. This two-firework setup allows us to run the
objective function and optimization on different computing resources, if
desired.

We also use an objective function with more than one objective. Note that
as long as we pass in the output vector to the spec (in the key "_y", as in the
basic example), we don't need to make any other changes to tell rocketsled the
objective function is multi-objective. Additionally, the objective function
has dimensions of differing data types (int, float, categorical), which is
automatically handled by rocketsled as long as the dimensions are passed into
MissionControl.configure(...).

Finally, we add some arguments to the MissionControl configuration before
launching.
"""

from fireworks import FireTaskBase, Firework, FWAction, LaunchPad, Workflow
from fireworks.core.rocket_launcher import rapidfire
from fireworks.utilities.fw_utilities import explicit_serialize

from rocketsled import MissionControl, OptTask

launchpad = LaunchPad(name="rsled")
opt_label = "opt_complex"
db_info = {"launchpad": launchpad, "opt_label": opt_label}
x_dim = [(16, 145), (0.0, 90.0), ["industry standard", "shark fin", "dolphin fin"]]


@explicit_serialize
class ComplexMultiObjTask(FireTaskBase):
    """
    An example of a complex, multiobjective task with directly competing
    objectives. The input vector is defined on a search space with numerical
    and categorical inputs.

    This task accepts a 3-vector of the form [int, float, str].
    """

    _fw_name = "ComplexMultiObjectiveTask"

    def run_task(self, fw_spec):
        x = fw_spec["_x"]
        fin_len = x[0]
        fin_angle = x[1]
        fin_type = x[2]
        cost = (14.1 * fin_len ** 1.847 + 12.0 + fin_angle * 100.0) / 1000.0
        drag = fin_angle ** 0.653 * float(fin_len) ** 1.2
        failure_prob = 0.5 - fin_len / 290 + (fin_angle ** 2.0) / 16200
        if fin_type == "shark fin":
            cost = cost * 1.05
            drag = drag * 1.15
            failure_prob = failure_prob * 0.75
        elif fin_type == "dolphin fin":
            cost = cost * 1.6
            drag = drag * 0.84
            failure_prob = failure_prob * 1.75
        return FWAction(update_spec={"_y": [cost, drag, failure_prob], "_x": x})


def wf_creator(x):
    """
    A workflow creator function returning a workflow of the following form:

                  simulation (fw1)
                      |
                optimization (fw2)

    Args:
        x ([list]): A 3 vector of the form [int, float, str], where the elements
            are constrained to the search space given in x_dim above.

    Returns:
        (Workflow): The workflow which will run the simulation and optimization
            fireworks.

    """
    simulation = Firework([ComplexMultiObjTask()], spec={"_x": x}, name="simulation")
    optimization = Firework([OptTask(**db_info)], name="optimization")
    return Workflow([simulation, optimization], {simulation: optimization})


def get_z(x):
    """
    An example function demonstrating how to use z_features.

    The get_z function should accept the same vector as the wf_creator (the x
    vector), and return all information that should be used for learning. If
    you want to use x for learning, make sure to include x in the returned
    z vector.

    Args:
        x ([list]): A 3 vector of the form [int, float, str], where the elements
            are constrained to the search space given in x_dim above.

    Returns:
        (list): The z vector, to be used for learning.

    """
    fin_len = x[0]
    fin_angle = x[1]
    useful_feature1 = fin_len + fin_angle ** 2
    useful_feature2 = fin_angle + fin_len
    return x + [useful_feature1, useful_feature2]


if __name__ == "__main__":
    # Make a MissionControl object
    mc = MissionControl(**db_info)

    # Reset the launchpad and optimization db for this example
    launchpad.reset(password=None, require_password=False)
    mc.reset(hard=True)

    # Configure the optimization db with MissionControl
    mc.configure(
        wf_creator=wf_creator,
        dimensions=x_dim,
        acq="maximin",
        predictor="GaussianProcessRegressor",
        get_z=get_z,
    )

    # Run 30 workflows + optimization
    launchpad.add_wf(wf_creator([100, 45.0, "dolphin fin"]))
    rapidfire(launchpad, nlaunches=30)

    # Examine and plot the optimization
    plt = mc.plot(print_pareto=True)
    plt.show()
