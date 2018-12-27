"""
A more complex example based on the advanced rocketsled tutorial.

The workflow creator function creates two Fireworks, one containing a mock 
simulation, and the other containing OptTask.

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
from rocketsled.examples.tasks import ComplexMultiObjTask

lpad = LaunchPad(name='rsled')
opt_label = "opt_complex"
dims = [(16, 145), (0.0, 90.0),
        ["industry standard", "shark fin", "dolphin fin"]]


def wf_creator(x):
    simulation = Firework([ComplexMultiObjTask()], spec={'_x': x},
                          name="simulation")
    optimization = Firework(
        [OptTask(wf_creator='rocketsled.examples.complex.wf_creator',
                 dimensions=dims,
                 lpad=lpad,
                 opt_label="opt_complex",
                 acq="maximin",
                 predictor="GaussianProcessRegressor",
                 get_z='rocketsled.examples.complex.get_z'
                 )],
        name="optimization")
    return Workflow([simulation, optimization], {simulation: optimization})


def get_z(x):
    fin_len = x[0]
    fin_angle = x[1]
    useful_feature1 = fin_len + fin_angle ** 2
    useful_feature2 = fin_angle + fin_len
    return x + [useful_feature1, useful_feature2]


if __name__ == "__main__":
    lpad.reset(password=None, require_password=False)
    lpad.add_wf(wf_creator([60, 45.0, "industry standard"]))
    rapidfire(lpad, nlaunches=500, sleep_time=0)
