# from __future__ import print_function

from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from fireworks.core.rocket_launcher import launch_rocket
from fireworks import FWAction, Workflow, Firework, LaunchPad
from turboworks.optimize import OptTask
from turboworks.db import DB
from turboworks.dummy import dummy_minimize
from matplotlib import pyplot as plot


@explicit_serialize
class CalculateTask(FireTaskBase):
    _fw_name = "CalculateTask"

    def run_task(self, fw_spec):

        A = fw_spec['A']
        B = fw_spec['B']
        C = fw_spec['C']

        output = {'_y': A**2 + B**2 / C}
        return FWAction(update_spec=output)

@explicit_serialize
class ArbitraryTask(FireTaskBase):
    _fw_name = "ArbitraryTask"

    def run_task(self, fw_spec):
        print("final task result:", fw_spec['_y'])

def get_x(z):
    return [z[1] * 2, z[2]**3]

def wf_creator(z):

    spec1 = {'A':z[0], 'B':z[1], 'C':z[2], '_z':z}
    Z_dim = [(1,5), (1,5), (1,5)]

    #CalculateTask writes _y field to the spec internally.

    firework1 = Firework([CalculateTask(), ArbitraryTask(),
                          OptTask(wf_creator ='test.wf_creator',
                                  get_x='test.get_x',
                                  # predictor='gp_minimize',
                                  dimensions=Z_dim)],
                         spec=spec1)

    return Workflow([firework1])

def example_predictor_wrapper(Z_ext, Y, Z_ext_dims):
    return dummy_minimize(Z_ext_dims)


if __name__ == "__main__":

    db = DB()
    db_meta = DB(collection = 'meta')
    db.nuke()
    db_meta.nuke()

    launchpad = LaunchPad()
    launchpad.reset('', require_password=False)
    launchpad.add_wf(wf_creator([1,4,3]))
    # launchpad.add_wf(wf_creator([3,5,1]))
    # launchpad.add_wf(wf_creator([2,5,1]))
    # launchpad.add_wf(wf_creator([5,5,1]))
    # launchpad.add_wf(wf_creator([2,4,3]))
    # launchpad.add_wf(wf_creator([1, 3, 3]))
    # launchpad.add_wf(wf_creator([3, 4, 1]))
    # launchpad.add_wf(wf_creator([2, 4, 1]))
    # launchpad.add_wf(wf_creator([5, 1, 1]))
    # launchpad.add_wf(wf_creator([2, 1, 3]))

    for i in range(127):
        launch_rocket(launchpad)





    # minima = []
    #
    # for i in range(100):
    #     launch_rocket(launchpad)
    #     minima.append(db.min.value)
    #
    #
    # plot.plot(range(len(minima)), minima)
    # plot.ylabel('Best Minimum Value')
    # plot.xlabel('Iteration')
    # plot.show()


    # check to see if issues with m_launch........
    # two z's can run at same time................

    # make sure no duplicate z/space searched exhaustively..............[done]
    # make module importable......................
    # make ability to start with x random guesses.
    # send anubhav code...........................
