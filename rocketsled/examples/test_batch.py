from __future__ import unicode_literals, print_function, division

from fireworks import Workflow, Firework, LaunchPad
from fireworks.core.rocket_launcher import rapidfire
from rocketsled.optimize import OptTask, random_guess
from rocketsled.examples.example_tasks import SumTask

opt_label = "opt_batch"
X_dim = [(1, 5), (1, 5), (1, 5)]


def wf_creator(x):
    spec = {'_x_opt': x}

    # CalculateTask writes _y_opt field to the spec internally.
    firework1 = Firework([SumTask(),
                          OptTask(wf_creator='rocketsled.examples.test_batch.'
                                             'wf_creator',
                                  dimensions=X_dim,
                                  host='localhost',
                                  port=27017,
                                  name='ROCKETSLED_EXAMPLES',
                                  opt_label=opt_label,
                                  batch_size=5)],
                         spec=spec)

    return Workflow([firework1])

def run_workflows():
    TESTDB_NAME = 'ROCKETSLED_EXAMPLES'
    launchpad = LaunchPad(name=TESTDB_NAME)

    # clean up tw database if necessary
    launchpad.reset(password=None, require_password=False)

    # load 10 batch workflows onto the launchpad
    for _ in range(10):
        launchpad.add_wf(wf_creator(random_guess(X_dim)))

    rapidfire(launchpad, nlaunches=10, sleep_time=0)

if __name__ == "__main__":
    run_workflows()