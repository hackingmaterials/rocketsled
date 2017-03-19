'''
Examples of using extra turboworks features.

Note that in this optimization we can keep all the data from the text_basic
example by using opt_label. Naming the opt_label parameter allows more than one
optimization to be stored and accessed.




'''

from fireworks.core.rocket_launcher import launch_rocket
from fireworks import Workflow, Firework, LaunchPad
from turboworks.optimize import OptTask
from turboworks.optdb import OptDB
from turboworks.utils import random_guess
from matplotlib import pyplot as plot
from turboworks_examples.calculate_task import MixedCalculateTask as CalculateTask





# use a wf_creator function with more arguments...
def wf_creator(z, my_kwarg=1):

    fw1_spec = {'A': z[0], 'B': z[1], 'C': z[2], 'D':z[3], '_z': z}
    fw1_dim = [(1,2),(1,2),(1,2), ("red", "green", "blue")]

    # CalculateTask writes _y field to the spec internally.

    firework1 = Firework([CalculateTask(),
                          OptTask(wf_creator='turboworks_examples.test_extras.wf_creator',
                                  dimensions=fw1_dim,
                                  get_x='turboworks_examples.test_extras.get_x',
                                  # predictor='gp_minimize',
                                  predictor = 'turboworks_examples.test_extras.example_predictor',
                                  opt_label="extras",
                                  duplicate_check=True,
                                  wf_creator_args={'my_kwarg':my_kwarg*2})],
                         spec=fw1_spec)

    return Workflow([firework1])


# An optional function which returns extra information 'x' from unique vector 'z'
def get_x(z):
    return [z[0] * 2, z[2]**3]

# how an example custom optimization function could be used
# replace the code inside example_predictor with your favorite optimizer

def example_predictor(Z_ext, Y, Z_ext_dims):
    # custom optimizer code goes here
    return random_guess(Z_ext_dims)


if __name__ == "__main__":

    launchpad = LaunchPad()
    opt_db = OptDB(opt_label="extras")  # only use database entries pertaining to this workflow
    opt_db.clean()   # only cleans previous runs of test_extras

    # optimization works with or without cleaning the test_basic run

    # uncomment the line below to reset fireworks
    # launchpad.reset('', require_password=False)

    example_data = [{'A': 1, 'B': 2, 'C': 1, 'D': 'blue', 'output': 19.121},
                    {'A': 2, 'B': 2, 'C': 2, 'D': 'red', 'output': 81.2},
                    {'A': 1, 'B': 1, 'C': 1, 'D': 'blue', 'output': 15.3}]

    launchpad.add_wf(wf_creator([1, 2, 1, "red"]))

    # store some precomputed data in the optimization db prior to running a workflow, if we desire
    opt_db.store(example_data, z_keys=['A', 'B', 'C', 'D'], y_key='output', opt_label='extras')

    # The number of loops to run
    # If >=21, the search space will be exhausted and OptTask will raise an exception.
    n_runs = 21

    minima = []
    for i in range(n_runs):
        launch_rocket(launchpad)
        minima.append(opt_db.min.value)
    plot.plot(range(len(minima)), minima)
    plot.ylabel('Best Minimum Value')
    plot.xlabel('Iteration')
    plot.show()
