from __future__ import print_function

from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from fireworks.core.rocket_launcher import launch_rocket
from fireworks import FWAction, Workflow, Firework, LaunchPad
from turboworks.vector_optimize import VectorOptimize
from turboworks.dummy_opt import dummy_minimize
from turboworks.manage_db import ManageDB


@explicit_serialize
class CalculateTask(FireTaskBase):
    _fw_name = "CalculateTask"

    def run_task(self, fw_spec):

        A = fw_spec['A']
        B = fw_spec['B']
        C = fw_spec['C']

        output = {'_y': A * B / C}
        return FWAction(update_spec=output)

@explicit_serialize
class ArbitraryTask(FireTaskBase):
    _fw_name = "ArbitraryTask"

    def run_task(self, fw_spec):
        print("final task result:", fw_spec['_y'])

def get_x(z):
    return [z[1] * .0000007, z[2]**0.87 - 15]

def wf_creator(z):

    spec1 = {'A':z[0], 'B':z[1], 'C':z[2], '_z':z}
    Z_dim = [(1.0,100.0), (1.0,200.0), (1.0,300.0)]

    firework1 = Firework([CalculateTask(), ArbitraryTask(),
                          VectorOptimize(wf_creator ='vector_optimize_test.wf_creator',
                                         get_x = 'vector_optimize_test.get_x',
                                         dimensions=Z_dim)],
                         spec=spec1, name='firework1')

    return Workflow([firework1])

if __name__ == "__main__":

    mdb = ManageDB()
    mdb.nuke()

    # set up the LaunchPad and reset it
    launchpad = LaunchPad()
    launchpad.reset('', require_password=False)
    launchpad.add_wf(wf_creator([1.0, 2.0, 3.0]))

    for i in range(300):
        launch_rocket(launchpad)
