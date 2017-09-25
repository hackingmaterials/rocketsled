"""
Benchmarking for turboworks infrastructure.
"""

from fireworks.core.rocket_launcher import launch_rocket
from fireworks import Workflow, Firework, LaunchPad, FWAction, FireTaskBase, explicit_serialize
from turboworks.optimize import OptTask, random_guess
from scipy.optimize import rosen
import pymongo
import time
from matplotlib import pyplot
import pickle
import numpy
import math

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"



@explicit_serialize
class BenchmarkTask(FireTaskBase):
    _fw_name = "BenchmarkTask"

    def run_task(self, fw_spec):

        # def schafferf7(x):
        #     fitness = 0
        #     normalizer = 1.0 / float(len(x) - 1)
        #     for i in range(len(x) - 1):
        #         si = math.sqrt(x[i] ** 2 + x[i + 1] ** 2)
        #         fitness += (normalizer * math.sqrt(si) * (math.sin(50 * si ** 0.20) + 1)) ** 2
        #     return fitness

        x = fw_spec['_x_opt']
        val = rosen(x)
        # val = schafferf7(x)
        # val = 10 * len(x) + sum([xi ** 2 - 10 * math.cos(xi * 2.0 * math.pi) for xi in x]) # rastrigin
        return FWAction(update_spec={'_y_opt': val})

# X_dim = [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)] # for rast
X_dim = [(-2.048, 2.048), (-2.048, 2.048), (-2.048, 2.048), (-2.048, 2.048), (-2.048, 2.048)] # for rosen
# X_dim = [(-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0)] # for schaffer
DB_NAME = 'rf'
COL_NAME = 'rosen'


# a workflow creator function which takes x and returns a workflow based on x
def wf_creator(x):

    spec = {'_x_opt':x}

    firework1 = Firework([BenchmarkTask(),
                          OptTask(wf_creator='benchmarks.bbfuns.wf_creator',
                                  dimensions=X_dim,
                                  host='localhost',
                                  port=27017,
                                  name=DB_NAME,
                                  opt_label=COL_NAME,
                                  predictor='RandomForestRegressor',
                                  n_search_points=1000)],
                          spec=spec)

    return Workflow([firework1])


if __name__ == "__main__":
    launchpad = LaunchPad(name=DB_NAME)

    collection = getattr(getattr(pymongo.mongo_client.MongoClient('localhost', 27017), DB_NAME), COL_NAME)

    # for _ in range(5):
    launchpad.reset(password=None, require_password=False)
    launchpad.add_wf(wf_creator(random_guess(X_dim)))

    min_vals = []
    for i in range(1000):
        launch_rocket(launchpad)
        cur = collection.find({'y': {'$exists': 1}}).sort('y', pymongo.ASCENDING).limit(1)
        for doc in cur:
            y = doc['y']
            print y, doc['x']
            min_vals.append(y)
            break


    # times = []
    # for i in iterations:
    #
    #     t_before = time.time()
    #     launch_rocket(launchpad)
    #     t_after = time.time()
    #     times.append(t_after - t_before)
    #
    # launchpad.connection.drop_database(TESTDB_NAME)
    # total_times.append(times)

    pickle.dump(min_vals, open('{}_{}.pickle'.format(DB_NAME, COL_NAME), 'wb'))

    pyplot.plot(list(range(len(min_vals))), min_vals)
    pyplot.show()


    # tear down database
    # launchpad.connection.drop_database(TESTDB_NAME)