from __future__ import unicode_literals, print_function, division

from fireworks.core.rocket_launcher import rapidfire
from fireworks import Workflow, Firework, LaunchPad
from turboworks.optimize import OptTask
from calculate_task import MixedCalculateTask as CalculateTask

opt_label = "opt_categorical"

def wf_creator(x):

    fw1_spec = {'A': x[0], 'B': x[1], 'C': x[2], 'D': x[3], '_x_opt': x}
    fw1_dim = [(1, 2), (1, 2), (1, 2), ("red", "green", "blue")]

    # CalculateTask writes _y_opt field to the spec internally.

    firework1 = Firework([CalculateTask(),
                          OptTask(wf_creator='examples.test_categorical.wf_creator',
                                  dimensions=fw1_dim,
                                  host='localhost',
                                  port=27017,
                                  name='turboworks',
                                  get_z='examples.test_categorical.get_z',
                                  duplicate_check=True,
                                  opt_label=opt_label)],
                         spec=fw1_spec)
    return Workflow([firework1])


def get_z(x):
    if x[1] == 1:
        cat = "tiger"
    else:
        cat = "lion"
    return [x[0]**2, cat]

def run_workflows(test_case=False):
    TESTDB_NAME = 'turboworks'
    launchpad = LaunchPad(name=TESTDB_NAME)
    if test_case:
        getattr(getattr(launchpad.connection, TESTDB_NAME), opt_label).drop()

    # clean up tw database if necessary
    launchpad.reset(password=None, require_password=False)

    launchpad.add_wf(wf_creator([1, 1, 2, "red"]))
    rapidfire(launchpad, nlaunches=23, sleep_time=0)

    # tear down database
    # launchpad.connection.drop_database(TESTDB_NAME)


if __name__ == "__main__":
    run_workflows()
