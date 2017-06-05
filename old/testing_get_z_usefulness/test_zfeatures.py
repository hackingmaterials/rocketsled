from fireworks.core.rocket_launcher import launch_rocket
from fireworks import Workflow, Firework, LaunchPad, FireTaskBase, FWAction
from fireworks.utilities.fw_utilities import explicit_serialize
from turboworks.optimize import OptTask
from pymongo import MongoClient
from matplotlib import pyplot
import numpy as np
import pickle


@explicit_serialize
class ABCTask(FireTaskBase):
    _fw_name = "ABCTask"

    def run_task(self, fw_spec):

        A = fw_spec['A']
        B = fw_spec['B']
        C = fw_spec['C']

        score = (A**2)*(B**2)/C
        output = {'_y_opt': score}
        return FWAction(update_spec=output)

def wf_creator(x, predictor, get_z, lpad):
    spec = {'A': x[0], 'B': x[1], 'C': x[2], '_x_opt': x}
    dim = [(1, 10), (1, 10), (1, 10)]

    firework = Firework([ABCTask(),
                        OptTask(wf_creator='turboworks_examples.test_zfeatures.wf_creator',
                                dimensions=dim,
                                lpad=lpad,
                                get_z=get_z,
                                predictor=predictor,
                                duplicate_check=True,
                                wf_creator_args=[predictor, get_z, lpad],
                                max=True,
                                opt_label='test_zfeatures',
                                n_points=1000)],
                        spec=spec)
    return Workflow([firework])

def get_z(x):
    A = x[0]
    B = x[1]
    C = x[2]
    return [A**2, B**2, A/C, B/C]


if __name__== "__main__":

    TESTDB_NAME = 'turboworks3'

    conn = MongoClient('localhost', 27017)
    db = getattr(conn, TESTDB_NAME)
    collection = db.test_zfeatures

    get_z = 'turboworks_examples.test_zfeatures.get_z'
    predictor = 'AdaBoostRegressor'
    name = '{}_withz'.format(predictor)

    n_runs = 50
    n_iterations = 50
    Y = []
    x = range(n_iterations)

    for i in range(n_runs):

        launchpad = LaunchPad(name=TESTDB_NAME)
        launchpad.reset(password=None, require_password=False)
        launchpad.add_wf(wf_creator([1, 1, 1], predictor, get_z, launchpad))

        y = []
        for _ in x:
            launch_rocket(launchpad)
            max_value = collection.find_one(sort=[("yi", -1)])["yi"]
            y.append(max_value)

        Y.append(y)
        launchpad.connection.drop_database(TESTDB_NAME)

    mean = np.mean(Y, axis=0)
    std = np.std(Y, axis=0)
    lower = [mean[i] - std[i] for i in range(len(mean))]
    upper = [mean[i] + std[i] for i in range(len(mean))]

    data = {'mean': mean, 'lower': lower, 'upper': upper}

    pickle.dump(data, open('{}.p'.format(name), 'w'))


