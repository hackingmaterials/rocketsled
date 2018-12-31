"""
Running a rocketsled optimization where the objective function has a categorical
argument.

--------------------------------------------------------------------------
The following workflow is only one Firework (one job), for example purposes.
However, FireWorks and rocketsled are capable of handling more complex
workflows including multiple jobs and advanced dependencies. Please see the
Fireworks and rocketsled documentation pages for more information:

https://hackingmaterials.github.io/rocketsled/
https://materialsproject.github.io/fireworks/
"""

from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.rocket_launcher import rapidfire
from fireworks import Workflow, Firework, LaunchPad, FireTaskBase, FWAction

from rocketsled import OptTask, MissionControl

launchpad = LaunchPad(name='rsled')
opt_label = "opt_complex"
db_info = {"launchpad": launchpad, "opt_label": opt_label}
x_dim = [(16, 145), (0.0, 90.0),
         ["industry standard", "shark fin", "dolphin fin"]]


@explicit_serialize
class ComplexMultiObjTask(FireTaskBase):
    """
    An example of a complex, multiobjective task (similar to MutliTask2/6) with
    directly competing objectives. The input vector is defined on a search
    space with numerical and categorical inputs.
    """
    _fw_name = "CMOT"

    def run_task(self, fw_spec):
        x = fw_spec['_x']
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
            failure_prob - failure_prob * 1.75

        return FWAction(update_spec={'_y': [cost, drag, failure_prob],
                                     '_x': x})


def wf_creator(x):
    simulation = Firework([ComplexMultiObjTask()], spec={'_x': x},
                          name="simulation")
    optimization = Firework([OptTask(**db_info)], name="optimization")
    return Workflow([simulation, optimization], {simulation: optimization})


def get_z(x):
    fin_len = x[0]
    fin_angle = x[1]
    useful_feature1 = fin_len + fin_angle ** 2
    useful_feature2 = fin_angle + fin_len
    return x + [useful_feature1, useful_feature2]


if __name__ == "__main__":
    mc = MissionControl(**db_info)
    mc.reset(hard=True)
    mc.configure(wf_creator=wf_creator,
                 dimensions=x_dim,
                 acq="maximin",
                 predictor="GaussianProcessRegressor",
                 get_z=get_z)
    launchpad.reset(password=None, require_password=False)
    launchpad.add_wf(wf_creator([100, 45.0, "dolphin fin"]))

    # Run 100 workflows + optimization
    rapidfire(launchpad, nlaunches=30)

    # Examine the optimization
    plt = mc.plot(print_pareto=True)
    plt.show()