from __future__ import print_function

from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from fireworks.core.rocket_launcher import launch_rocket
from fireworks import FWAction, Workflow, Firework, LaunchPad
from turboworks.optimize import OptTask
from turboworks.db import DB
from turboworks.dummy import dummy_minimize


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
    Z_dim = [(1.0,100.0), (1.0,100.0), (1.0,100.0)]

    #CalculateTask writes _y field to the spec internally.

    firework1 = Firework([CalculateTask(), ArbitraryTask(),
                          OptTask(wf_creator ='vector_optimize_test.wf_creator',
                                  get_x='vector_optimize_test.get_x',
                                  predictor='gp_minimize',
                                  dimensions=Z_dim)],
                         spec=spec1)

    return Workflow([firework1])

def example_predictor_wrapper(Z_ext, Y, Z_ext_dims):
    return dummy_minimize(Z_ext_dims)


if __name__ == "__main__":

    db = DB()
    db.nuke()

    launchpad = LaunchPad()
    launchpad.reset('', require_password=False)
    launchpad.add_wf(wf_creator([15.0, 25.0, 35.0]))

    for i in range(100):
        launch_rocket(launchpad)

    from pprint import pprint
    pprint(db.min.data)