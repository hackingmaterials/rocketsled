"""
Benchmarking for turboworks infrastructure.
"""

from fireworks.core.rocket_launcher import launch_rocket
from fireworks import Workflow, Firework, LaunchPad
from rocketsled.optimize import OptTask
from examples.calculate_task import BasicCalculateTask as CalculateTask
import time, pickle
from matplotlib import pyplot
import numpy

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"


# a workflow creator function which takes x and returns a workflow based on x
def wf_creator(x):

    spec = {'A':x[0], 'B':x[1], 'C':x[2], '_x_opt':x}
    X_dim = [(1, 1000), (1, 1000), (1, 1000)]

    # CalculateTask writes _y_opt field to the spec internally.

    firework1 = Firework([CalculateTask(),
                          OptTask(wf_creator='benchmarks.no_opt.wf_creator',
                                  dimensions=X_dim,
                                  host='localhost',
                                  port=27017,
                                  name='turboworks',
                                  opt_label='no_opt',
                                  predictor='random_guess')],
                          spec=spec)

    return Workflow([firework1])



if __name__ == "__main__":
    TESTDB_NAME = 'turboworks'
    launchpad = LaunchPad(name=TESTDB_NAME)

    total_times = []
    iterations = list(range(1000))

    for _ in range(100):
        launchpad.reset(password=None, require_password=False)
        launchpad.add_wf(wf_creator([500, 500, 500]))

        times = []
        for i in iterations:

            t_before = time.time()
            launch_rocket(launchpad)
            t_after = time.time()
            times.append(t_after - t_before)

        launchpad.connection.drop_database(TESTDB_NAME)
        total_times.append(times)



    avg = numpy.mean(total_times, axis=0)
    pickle.dump({'iterations': iterations[1:], 'avg': avg[1:], 'total_times': total_times},
                open('no_opt.pickle', 'wb'))

    pyplot.plot(iterations[1:], avg[1:])
    pyplot.show()


    # tear down database
    # launchpad.connection.drop_database(TESTDB_NAME)



