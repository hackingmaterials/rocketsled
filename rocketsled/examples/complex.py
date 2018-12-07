from __future__ import unicode_literals, print_function, unicode_literals

"""
A more complex example based on the advanced rocketsled tutorial.

The workflow creator function creates two Fireworks, one containing a mock 
simulation, and the other containing OptTask. 
"""

from fireworks.core.rocket_launcher import rapidfire
from fireworks import Workflow, Firework, LaunchPad
from rocketsled import OptTask
from rocketsled.examples.tasks import ComplexMultiObjTask


def wf_creator(x):
    X_dim = [(16, 145), (0.0, 90.0),
             ["industry standard", "shark fin", "dolphin fin"]]
    simulation = Firework([ComplexMultiObjTask()], spec={'_x': x},
                          name="simulation")
    optimization = Firework(
        [OptTask(wf_creator='rocketsled.examples.complex.wf_creator',
                 dimensions=X_dim,
                 host='localhost',
                 port=27017,
                 opt_label="opt_complex",
                 acq="maximin",
                 predictor="GaussianProcessRegressor",
                 get_z='rocketsled.examples.complex.get_z',
                 name='rsled')],
        name="optimization")
    return Workflow([simulation, optimization], {simulation: optimization})

def get_z(x):
    fin_len = x[0]
    fin_angle = x[1]
    useful_feature1 = fin_len + fin_angle ** 2
    useful_feature2 = fin_angle + fin_len
    return x + [useful_feature1, useful_feature2]

def run_workflows():
    TESTDB_NAME = 'rsled'
    launchpad = LaunchPad(name=TESTDB_NAME)
    launchpad.reset(password=None, require_password=False)
    launchpad.add_wf(wf_creator([60, 45.0, "industry standard"]))
    rapidfire(launchpad, nlaunches=500, sleep_time=0)

    # tear down database
    # launchpad.connection.drop_database(TESTDB_NAME)

if __name__ == "__main__":
    run_workflows()



