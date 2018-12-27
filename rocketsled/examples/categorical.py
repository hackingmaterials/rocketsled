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

from fireworks.core.rocket_launcher import rapidfire
from fireworks import Workflow, Firework, LaunchPad

from rocketsled import OptTask
from rocketsled.examples.tasks import MixedCalculateTask

opt_label = "opt_categorical"
lpad = LaunchPad(name='rsled')
dims = [(1, 2), (1, 2), (1, 2), ("red", "green", "blue")]


def wf_creator(x):
    fw1_spec = {'A': x[0], 'B': x[1], 'C': x[2], 'D': x[3], '_x': x}

    # CalculateTask writes _y field to the spec internally.

    firework1 = Firework([MixedCalculateTask(),
                          OptTask(wf_creator='rocketsled.examples.categorical.'
                                             'wf_creator',
                                  dimensions=dims,
                                  lpad=lpad,
                                  get_z='rocketsled.examples.categorical.'
                                        'get_z',
                                  duplicate_check=True,
                                  opt_label=opt_label)],
                         spec=fw1_spec)
    return Workflow([firework1])


def get_z(x):
    if x[1] == 1:
        cat = "tiger"
    else:
        cat = "lion"
    return [x[0] ** 2, cat]


if __name__ == "__main__":
    lpad.reset(password=None, require_password=False)
    lpad.add_wf(wf_creator([1, 1, 2, "red"]))
    rapidfire(lpad, nlaunches=23, sleep_time=0)
