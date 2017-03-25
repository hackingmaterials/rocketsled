'''
Examples of using extra turboworks features.

Note that in this optimization we can keep all the data from the text_basic
example by using opt_label. Naming the opt_label parameter allows more than one
optimization to be stored and accessed.
'''

from fireworks.core.rocket_launcher import launch_rocket, rapidfire
from fireworks import Workflow, Firework, LaunchPad
from turboworks.optimize import OptTask
from turboworks.utils import random_guess
from calculate_task import MixedCalculateTask as CalculateTask



# use a wf_creator function with more arguments...
def wf_creator(z, my_kwarg=1):

    fw1_spec = {'A': z[0], 'B': z[1], 'C': z[2], 'D':z[3], '_z': z}
    fw1_dim = [(1,2),(1,2),(1,2), ("red", "green", "blue")]

    # CalculateTask writes _y field to the spec internally.

    firework1 = Firework([CalculateTask(),
                          OptTask(wf_creator='turboworks_examples.test_extras.wf_creator',
                                  dimensions=fw1_dim,
                                  get_x='turboworks_examples.test_extras.get_x',
                                  # predictor='gp_minimize',  # use one of the 4 built-in optimizers
                                  predictor = 'turboworks_examples.test_extras.example_predictor',  # or your own
                                  wf_creator_args={'my_kwarg': my_kwarg * 2},
                                  duplicate_check=True,
                                  opt_label="opt_extras_example")],
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

    TESTDB_NAME = 'turboworks'
    launchpad = LaunchPad(name=TESTDB_NAME)
    launchpad.reset(password=None, require_password=False)
    launchpad.add_wf(wf_creator([1, 1, 2, "red"]))


    # if n_launches > 23 for this particular example, the search space will be exhausted and OptTask will throw
    # an exception
    rapidfire(launchpad, nlaunches=23, sleep_time=0)