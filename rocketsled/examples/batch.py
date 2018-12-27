"""
An example of running optimizations in batches. For example, submitting 5
workflows, running them, optimizing, and then submitting the next best 5 
workflows to the LPAD. 
"""

from fireworks import Workflow, Firework, LaunchPad
from fireworks.core.rocket_launcher import rapidfire

from rocketsled import OptTask
from rocketsled.utils import random_guess
from rocketsled.examples.tasks import SumTask

LPAD = LaunchPad(name='rsled')
opt_label = "opt_batch"
X_dim = [(1, 5), (1, 5), (1, 5)]


def wf_creator(x):
    spec = {'_x': x}

    # CalculateTask writes _y field to the spec internally.
    firework1 = Firework([SumTask(),
                          OptTask(wf_creator='rocketsled.examples.batch.'
                                             'wf_creator',
                                  dimensions=X_dim,
                                  lpad=LPAD,
                                  opt_label=opt_label,
                                  batch_size=5)],
                         spec=spec)

    return Workflow([firework1])


if __name__ == "__main__":
    # clean up database if necessary
    LPAD.reset(password=None, require_password=False)

    # load 10 batch workflows onto the launchpad
    for _ in range(5):
        LPAD.add_wf(wf_creator(random_guess(X_dim)))

    rapidfire(LPAD, nlaunches=0, sleep_time=0)