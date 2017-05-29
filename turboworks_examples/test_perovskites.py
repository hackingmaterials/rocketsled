from fireworks.core.rocket_launcher import launch_rocket
from fireworks import Workflow, Firework, LaunchPad, FireTaskBase, FWAction
from fireworks.utilities.fw_utilities import explicit_serialize
from turboworks.optimize import OptTask, random_guess
from pymongo import MongoClient
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
import pandas as pd
import numpy as np
import pickle



# 20 light splitters in terms of atomic number
good_cands_ls = [(3, 23, 0), (11, 51, 0), (12, 73, 1), (20, 32, 0), (20, 50, 0), (20, 73, 1), (38, 32, 0), (38, 50, 0),
                 (38, 73, 1), (39, 73, 2), (47, 41, 0), (50, 22, 0), (55, 41, 0), (56, 31, 4), (56, 49, 4), (56, 50, 0),
                 (56, 73, 1), (57, 22, 1), (57, 73, 2), (82, 31, 4)]

# 8 oxide shields in terms of atomic number
good_cands_os = [(20, 50, 0), (37, 22, 4), (37, 41, 0), (38, 22, 0), (38, 31, 4), (38, 50, 0), (55, 73, 0),
                 (56, 49, 4)]
n_cands = 18928

# Names (for categorical)
ab_names = ['Li', 'Be', 'B', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
               'Zn', 'Ga', 'Ge', 'As', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
               'Sb', 'Te', 'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']
c_names = ['O3', 'O2N', 'ON2', 'N3', 'O2F', 'OFN', 'O2S']

# Atomic
ab_atomic =[3, 4, 5, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 37, 38, 39, 40, 41, 42,
            44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 56, 57, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83]
c_atomic = list(range(7))

# Mendeleev number
ab_mend = [1, 67, 72, 2, 68, 73, 78, 3, 7, 11, 43, 46, 49, 52, 55, 58, 61, 64, 69, 74, 79, 84, 4, 8, 12, 44, 47, 50, 56,
           59, 62, 65, 70, 75, 80, 85, 90, 5, 9, 13, 45, 48, 51, 54, 57, 60, 63, 66, 71, 76, 81, 86]
c_mend  = [261, 256, 251, 246, 267, 262, 262]

# Mendeleev rank
ab_mendrank = [sorted(ab_mend).index(i) for i in ab_mend]
c_mendrank = [3, 2, 1, 0, 6, 4, 5]

# Corrected and relevant perovskite data
perovskites = pd.read_csv('unc.csv')


@explicit_serialize
class EvaluateFitnessTask(FireTaskBase):
    _fw_name = "EvaluateFitnessTask"

    def run_task(self, fw_spec):

        # mendeleev ranked params
        a_mr = fw_spec['A']
        b_mr = fw_spec['B']
        c_mr = fw_spec['C']

        # convert from mendeleev rank and score compound
        a_i = ab_mendrank.index(a_mr)
        b_i = ab_mendrank.index(b_mr)
        c_i = c_mendrank.index(c_mr)
        a = ab_names[a_i]
        b = ab_names[b_i]
        c = c_names[c_i]

        data = perovskites.loc[(perovskites['A'] == a) & (perovskites['B'] == b) & (perovskites['anion'] == c)]
        score = float(data['complex_score'])
        output = {'_y_opt': score}
        return FWAction(update_spec=output)

dim = [(0, 51), (0, 51), (0, 6)]

def wf_creator(x, predictor, get_z, lpad):
    spec = {'A': x[0], 'B': x[1], 'C': x[2], '_x_opt': x}

    firework = Firework([EvaluateFitnessTask(),
                        OptTask(wf_creator='turboworks_examples.test_perovskites.wf_creator',
                                dimensions=dim,
                                lpad=lpad,
                                # get_z=get_z,
                                predictor=predictor,
                                duplicate_check=True,
                                wf_creator_args=[predictor, get_z, lpad],
                                max=True,
                                opt_label='test_perovskites',
                                n_search_points=1000,
                                n_train_points=1000)],
                        spec=spec)
    return Workflow([firework])


# api_key = "ya1iJA4H8O6TLGut"
# def get_z(x):


if __name__ =="__main__":

    TESTDB_NAME = 'perovskites1'
    predictor = 'RandomForestRegressor'
    get_z = 'turboworks_examples.test_perovskites.get_z'
    n_iterations = 5
    n_runs = 2
    filename = 'perovskites_{}_noz_{}iters_{}runs.p'.format(predictor, n_iterations, n_runs)

    conn = MongoClient('localhost', 27017)
    db = getattr(conn, TESTDB_NAME)
    collection = db.test_perovskites

    Y = []
    for i in range(n_runs):

        launchpad = LaunchPad(name=TESTDB_NAME)
        launchpad.reset(password=None, require_password=False)
        launchpad.add_wf(wf_creator(random_guess(dim), predictor, get_z, launchpad))

        y = []
        x = range(n_iterations)
        for _ in range(n_iterations):
            launch_rocket(launchpad)
            n_cands = collection.find({'yi':30.0}).count()
            y.append(n_cands)

        Y.append(y)
        launchpad.connection.drop_database(TESTDB_NAME)

    # mean = np.mean(Y, axis=0)
    # std = np.std(Y, axis=0)
    # lower = [mean[i] - std[i] for i in range(len(mean))]
    # upper = [mean[i] + std[i] for i in range(len(mean))]

    data = {'Y': Y}
    pickle.dump(data, open(filename, 'w'))