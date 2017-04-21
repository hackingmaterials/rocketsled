'''
Examples of using extra turboworks features.

In this task, the calculation and optimization includes a categorical dimension, a function to fetch extra features
(get_z), a custom predictor function, extra arguments to the workflow creator, duplicate checking enabled, and a custom
storage location for the optimization data.
Also, a demo of how to use the lpad kwarg to store optimization data based on a Firework's LaunchPad object.
'''

from fireworks.core.rocket_launcher import launch_rocket, rapidfire
from fireworks import Workflow, Firework, LaunchPad
from turboworks.optimize import OptTask, random_guess
from calculate_task import MixedCalculateTask as CalculateTask


# use a wf_creator function with more arguments...
def wf_creator(x, launchpad, my_arg, my_kwarg=1):

    fw1_spec = {'A': x[0], 'B': x[1], 'C': x[2], 'D':x[3], '_x_opt': x}
    fw1_dim = [(1,2),(1,2),(1,2), ("red", "green", "blue")]

    # CalculateTask writes _y_opt field to the spec internally.

    firework1 = Firework([CalculateTask(),
                          OptTask(wf_creator='turboworks_examples.test_extras.wf_creator',
                                  dimensions=fw1_dim,
                                  get_z='turboworks_examples.test_extras.get_z',
                                  # predictor='gp_minimize',  # use one of the 4 built-in optimizers
                                  predictor = 'turboworks_examples.test_extras.example_predictor',  # or your own
                                  max = True, # find the maximum value instead of the minimum
                                  lpad = launchpad,
                                  wf_creator_args = [launchpad, my_arg * 3],
                                  wf_creator_kwargs={'my_kwarg': my_kwarg * 2},
                                  duplicate_check=True,
                                  opt_label="opt_extras_example")],
                         spec=fw1_spec)

    return Workflow([firework1])


# An optional function which returns extra information 'z' from unique vector 'x'
def get_z(x):
    return [x[0] * 2, x[2] ** 3]

# how an example custom optimization function could be used
# replace the code inside example_predictor with your favorite optimizer

def example_predictor(X_tot, Y, X_tot_dims):
    # custom optimizer code goes here
    return random_guess(X_tot_dims)


if __name__ == "__main__":

    TESTDB_NAME = 'turboworks'
    launchpad = LaunchPad(name=TESTDB_NAME)
    launchpad.reset(password=None, require_password=False)
    launchpad.add_wf(wf_creator([1, 1, 2, "red"], launchpad, 3, my_kwarg= 1))


    # if n_launches > 23 for this particular example, the search space will be exhausted and OptTask will throw
    # an exception
    rapidfire(launchpad, nlaunches=23, sleep_time=0)

    # tear down database
    # launchpad.connection.drop_database(TESTDB_NAME)