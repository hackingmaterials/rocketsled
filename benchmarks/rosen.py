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

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"



@explicit_serialize
class RosenTask(FireTaskBase):
    _fw_name = "RosenTask"

    def run_task(self, fw_spec):

        x = fw_spec['_x_opt']
        val = rosen(x)
        return FWAction(update_spec={'_y_opt': val})

X_dim = [(0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0), (0.0, 2.0)]


# a workflow creator function which takes x and returns a workflow based on x
def wf_creator(x):

    spec = {'_x_opt':x}

    firework1 = Firework([RosenTask(),
                          OptTask(wf_creator='benchmarks.rosen.wf_creator',
                                  dimensions=X_dim,
                                  host='localhost',
                                  port=27017,
                                  name='rf_turboworks_rosen',
                                  opt_label='rosen',
                                  # predictor='GaussianProcessRegressor',
                                  n_search_points=1000)],
                          spec=spec)

    return Workflow([firework1])


if __name__ == "__main__":
    TESTDB_NAME = 'rf_turboworks_rosen'
    launchpad = LaunchPad(name=TESTDB_NAME)

    collection = pymongo.mongo_client.MongoClient('localhost', 27017).rf_turboworks_rosen.rosen


    # for _ in range(5):
    launchpad.reset(password=None, require_password=False)
    launchpad.add_wf(wf_creator(random_guess(X_dim)))

    min_vals = []
    for i in range(1000):
        print "running this"
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

    pickle.dump(min_vals, open('rosen_rf.pickle', 'wb'))

    pyplot.plot(list(range(len(min_vals))), min_vals)
    pyplot.show()


    # tear down database
    # launchpad.connection.drop_database(TESTDB_NAME)