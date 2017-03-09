"""
Running examples to see how turboworks works and/or fails.
"""

from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from fireworks.core.rocket_launcher import launch_rocket
from fireworks import FWAction, Workflow, Firework, LaunchPad
from turboworks.optimize import OptTask
from turboworks.db import DB
from turboworks.dummy import dummy_minimize
from matplotlib import pyplot as plot

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"


dims = [(1,5), (1,5), (1,5)]


# Some arbitrary tasks
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



# An optional function which returns extra information 'x' from unique vector 'z'
def get_x(z):
    return [z[1] * 2, z[2]**3]


# a workflow creator function which takes z and returns a workflow based on z
def wf_creator(z):

    spec1 = {'A':z[0], 'B':z[1], 'C':z[2], '_z':z}
    Z_dim = dims

    #CalculateTask writes _y field to the spec internally.

    firework1 = Firework([CalculateTask(), ArbitraryTask(),
                          OptTask(wf_creator ='test.wf_creator',
                                  get_x='test.get_x',
                                  predictor='example_predictor_wrapper',
                                  dimensions=Z_dim)],
                         spec=spec1)

    return Workflow([firework1])


# how an example custom optimization function could be used
# replace dummy_mimize with your favorite optimizer

def example_predictor_wrapper(Z_ext, Y, Z_ext_dims):
    return dummy_minimize(Z_ext_dims)

# test a serial implementation of turboworks
def test_serial(n_launches):
    launchpad.add_wf(wf_creator([1,2,1]))

    minima = []
    for i in range(n_launches):
        launch_rocket(launchpad)
        minima.append(db.min.value)

    plot.plot(range(len(minima)), minima)
    plot.ylabel('Best Minimum Value')
    plot.xlabel('Iteration')
    plot.show()

# try a parallel implementation of turboworks
def test_parallel(n_processes):
    # after running this function, from terminal use rlaunch multi n_processes
    for i in range(n_processes):
        launchpad.add_wf(wf_creator(dummy_minimize(dims)))


if __name__ == "__main__":

    db = DB()
    db.reset()
    launchpad = LaunchPad()
    # uncomment the line below to run
    # launchpad.reset('', require_password=False)

    # test_serial(20)
    test_parallel(5)




