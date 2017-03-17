'''
Dealing with multiple optimizations in one loop.
'''


from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from fireworks.core.rocket_launcher import launch_rocket
from fireworks import FWAction, Workflow, Firework, LaunchPad
from turboworks.optimize import OptTask
from turboworks.db import DB
from turboworks.dummy import dummy_minimize
from matplotlib import pyplot as plot


# Some arbitrary tasks
@explicit_serialize
class CalculateTask(FireTaskBase):
    _fw_name = "CalculateTask1"

    def run_task(self, fw_spec):

        A = fw_spec['A']
        B = fw_spec['B']
        C = fw_spec['C']
        D = fw_spec['D']

        score = A**2 + B**2 / C
        score += 30 if D == 'red' else 0
        score -= 20 if D == 'green' else 0

        output = {'_y': score}
        return FWAction(update_spec=output)


def wf_creator(z, my_kwarg=1):


    fw1_spec = {'A': z[0], 'B': z[1], 'C': z[2], 'D':z[3], '_z': z}
    fw1_dim = [(1,2),(1,2),(1,2), ("red", "green", "blue")]

    # CalculateTask writes _y field to the spec internally.

    firework1 = Firework([CalculateTask(),
                          OptTask(wf_creator='test_extras.wf_creator',
                                  dimensions=fw1_dim,
                                  get_x='test_extras.get_x',
                                  predictor='gp_minimize',
                                  opt_label="firework1",
                                  duplicate_check=True,
                                  wf_creator_args={'my_kwarg':my_kwarg*2})],
                         spec=fw1_spec)

    return Workflow([firework1])


# An optional function which returns extra information 'x' from unique vector 'z'
def get_x(z):
    return [z[0] * 2, z[2]**3]

# how an example custom optimization function could be used
# replace dummy_mimize with your favorite optimizer
def example_predictor(Z_ext, Y, Z_ext_dims):

    # custom optimizer code goes here
    return dummy_minimize(Z_ext_dims)


# # try a parallel implementation of turboworks
# def test_parallel(n_processes):
#     # after running this function, from terminal use rlaunch multi n_processes
#     for i in range(n_processes):
#         launchpad.add_wf(wf_creator(dummy_minimize(dims)))


# test a serial implementation of turboworks
def test_serial(n_launches):

    minima = []
    for i in range(n_launches):
        launch_rocket(launchpad)
        minima.append(db.min.value)
    plot.plot(range(len(minima)), minima)
    plot.ylabel('Best Minimum Value')
    plot.xlabel('Iteration')
    plot.show()


if __name__ == "__main__":
    launchpad = LaunchPad()
    db = DB()
    db.clean()
    launchpad.add_wf(wf_creator([1, 2, 1, "red"]))

    # put some precomputed data in the optimization db prior to running a workflow, if we desire

    some_data = [{'A': 1, 'B': 2, 'C': 1, 'D': 'blue', 'output': 19.121},
                 {'A': 2, 'B': 2, 'C': 2, 'D': 'red', 'output': 81.2},
                 {'A': 1, 'B': 1, 'C': 1, 'D': 'blue', 'output': 15.3}]

    db.process_and_store(some_data, z_keys=['A', 'B', 'C', 'D'], y_key='output')

    test_serial(1)
