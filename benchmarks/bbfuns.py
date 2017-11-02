"""
Benchmarking for turboworks infrastructure.
"""

from fireworks.core.rocket_launcher import launch_rocket
from fireworks import Workflow, Firework, LaunchPad, FWAction, FireTaskBase, explicit_serialize
from rocketsled.optimize import OptTask, random_guess
from scipy.optimize import rosen
import pymongo
import time
from matplotlib import pyplot
import pickle
import numpy as np
import math

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"

OPT_NAME = 'ran'
BBF_NAME = 'dummy'

@explicit_serialize
class BenchmarkTask(FireTaskBase):
    _fw_name = "BenchmarkTask"

    def run_task(self, fw_spec):


        def schafferf7(bx):
            fitness = 0
            normalizer = 1.0 / float(len(bx) - 1)
            for i in range(len(bx) - 1):
                si = math.sqrt(bx[i] ** 2 + bx[i + 1] ** 2)
                fitness += (normalizer * math.sqrt(si) * (math.sin(50 * si ** 0.20) + 1)) ** 2
            return fitness

        def rastrigin(bx):
            return 10 * len(bx) + sum([xi ** 2 - 10 * math.cos(xi * 2.0 * math.pi) for xi in bx]) # rastrigin

        def dummy(bx):
            return (bx[0] + bx[1] + bx[2]) / (bx[3] + bx[4])

        x = fw_spec['_x_opt']

        funmap = {'rast': rastrigin, 'schaf': schafferf7, 'dummy': dummy, 'rosen': rosen}

        val = funmap[BBF_NAME](x)
        return FWAction(update_spec={'_y_opt': val})

rast_dim = [(-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12), (-5.12, 5.12)] # for rast
rosen_dim = [(-2.048, 2.048), (-2.048, 2.048), (-2.048, 2.048), (-2.048, 2.048), (-2.048, 2.048)] # for rosen
schaf_dim = [(-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0)] # for schaffer
dummy_dim = [(1.0, 100.0), (1.0, 100.0), (1.0, 100.0), (1.0, 100.0), (1.0, 100.0)] # for dummy

dimmap = {'rosen': rosen_dim, 'rast': rast_dim, 'schaf': schaf_dim, 'dummy': dummy_dim}
predmap = {'gp': 'GaussianProcessRegressor', 'rf': 'RandomForestRegressor', 'ran': 'random_guess', 'gbt': 'GradientBoostingRegressor'}

# a workflow creator function which takes x and returns a workflow based on x
def wf_creator(x):

    spec = {'_x_opt':x}

    firework1 = Firework([BenchmarkTask(),
                          OptTask(wf_creator='benchmarks.bbfuns.wf_creator',
                                  dimensions=dimmap[BBF_NAME],
                                  host='localhost',
                                  port=27017,
                                  name=OPT_NAME,
                                  opt_label=BBF_NAME,
                                  predictor=predmap[OPT_NAME],
                                  n_search_points=10000)],
                          spec=spec)

    return Workflow([firework1])


if __name__ == "__main__":
    launchpad = LaunchPad(name=OPT_NAME)

    collection = getattr(getattr(pymongo.mongo_client.MongoClient('localhost', 27017), OPT_NAME), BBF_NAME)

    # for _ in range(5):
    launchpad.reset(password=None, require_password=False)
    launchpad.add_wf(wf_creator(random_guess(dimmap[BBF_NAME])))

    total_min_vals = []
    for j in range(10):
        min_vals = []
        for i in range(100):
            launch_rocket(launchpad)
            cur = collection.find({'y': {'$exists': 1}}).sort('y', pymongo.ASCENDING).limit(1)
            for doc in cur:
                y = doc['y']
                print y, doc['x']
                min_vals.append(y)
                break
        total_min_vals.append(min_vals)
        collection.drop()

    avg_min_vals = np.mean(total_min_vals, axis=0)

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

    pickle.dump(total_min_vals, open('{}_{}.pickle'.format(OPT_NAME, BBF_NAME), 'wb'))

    pyplot.plot(list(range(len(avg_min_vals))), avg_min_vals)
    print '{} {} {}'.format(OPT_NAME, BBF_NAME, avg_min_vals[-1])
    pyplot.show()


    # tear down database
    # launchpad.connection.drop_database(TESTDB_NAME)