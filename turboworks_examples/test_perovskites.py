from fireworks.core.rocket_launcher import launch_rocket
from fireworks import Workflow, Firework, LaunchPad, FireTaskBase, FWAction
from fireworks.utilities.fw_utilities import explicit_serialize
from turboworks.optimize import OptTask, random_guess
from pymongo import MongoClient
from matminer.descriptors.composition_features import get_pymatgen_descriptor
from pymatgen import Element
import pandas as pd
import numpy as np
import pickle


# 20 solar water splitter perovskite candidates in terms of atomic number
good_cands_ls = [(3, 23, 0), (11, 51, 0), (12, 73, 1), (20, 32, 0), (20, 50, 0), (20, 73, 1), (38, 32, 0), (38, 50, 0),
                 (38, 73, 1), (39, 73, 2), (47, 41, 0), (50, 22, 0), (55, 41, 0), (56, 31, 4), (56, 49, 4), (56, 50, 0),
                 (56, 73, 1), (57, 22, 1), (57, 73, 2), (82, 31, 4)]

# 8 oxide shields in terms of atomic number
good_cands_os = [(20, 50, 0), (37, 22, 4), (37, 41, 0), (38, 22, 0), (38, 31, 4), (38, 50, 0), (55, 73, 0),
                 (56, 49, 4)]
cands = 18928

# Names (for categorical)
ab_names = ['Li', 'Be', 'B', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
               'Zn', 'Ga', 'Ge', 'As', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
               'Sb', 'Te', 'Cs', 'Ba', 'La', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi']
c_names = ['O3', 'O2N', 'ON2', 'N3', 'O2F', 'OFN', 'O2S']

# Atomic
ab_atomic2 = [Element(name).Z for name in ab_names]
c_atomic = list(range(7))

# Mendeleev number
# ab_mend = [1, 67, 72, 2, 68, 73, 78, 3, 7, 11, 43, 46, 49, 52, 55, 58, 61, 64, 69, 74, 79, 84, 4, 8, 12, 44, 47, 50, 56,
#            59, 62, 65, 70, 75, 80, 85, 90, 5, 9, 13, 45, 48, 51, 54, 57, 60, 63, 66, 71, 76, 81, 86]
# c_mend  = [261, 256, 251, 246, 267, 262, 262]
ab_mend = [Element(name).mendeleev_no for name in ab_names]
c_mend = [np.sum(get_pymatgen_descriptor(anion, 'mendeleev_no')) for anion in c_names]

# Mendeleev rank
ab_mendrank = [sorted(ab_mend).index(i) for i in ab_mend]
c_mendrank = [4, 3, 2, 1, 6, 5, 0]

# Corrected and relevant perovskite data
perovskites = pd.read_csv('unc.csv')


def mend_to_name(a_mr, b_mr, c_mr):
    # go from mendeleev rank to name
    a_i = ab_mendrank.index(a_mr)
    b_i = ab_mendrank.index(b_mr)
    c_i = c_mendrank.index(c_mr)
    a = ab_names[a_i]
    b = ab_names[b_i]
    c = c_names[c_i]
    return a, b, c


@explicit_serialize
class EvaluateFitnessTask(FireTaskBase):
    _fw_name = "EvaluateFitnessTask"

    def run_task(self, fw_spec):

        # mendeleev ranked params
        a_mr = fw_spec['A']
        b_mr = fw_spec['B']
        c_mr = fw_spec['C']

        # convert from mendeleev rank and score compound
        a, b, c = mend_to_name(a_mr, b_mr, c_mr)

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
                                get_z=get_z,
                                predictor=predictor,
                                duplicate_check=True,
                                wf_creator_args=[predictor, get_z, lpad],
                                max=True,
                                opt_label='test_perovskites',
                                n_search_points=20000,
                                n_train_points=20000,
                                n_generation_points=1000)],
                        spec=spec)
    return Workflow([firework])


def get_z(x):
    descriptors = ['X', 'average_ionic_radius']
    # descriptors = ['average_ionic_radius']
    a, b, c = mend_to_name(x[0], x[1], x[2])
    name = a + b + c
    conglomerate = [get_pymatgen_descriptor(name, d) for d in descriptors]
    means = [np.mean(k) for k in conglomerate]
    stds = [np.std(k) for k in conglomerate]
    ranges = [np.ptp(k) for k in conglomerate]
    z = means + stds + ranges
    return z


if __name__ =="__main__":
    # using all 6 features: ...(no extra name extension)
    # using no features: has 'noz' in title
    # using just electroneg features ...eneg
    # using just average ionic radius features ...air

    TESTDB_NAME = 'wiz'
    predictor = 'RandomForestRegressor'
    get_z = 'turboworks_examples.test_perovskites.get_z'
    # n_iterations = 5000
    n_cands = 20
    n_runs = 20
    # filename = 'perovskites_{}_withz_{}iters_{}runs.p'.format(predictor, n_iterations, n_runs)
    filename = 'perovskites_{}_withz_{}cands_{}runs.p'.format(predictor, n_cands, n_runs)

    Y = []
    for i in range(n_runs):
        rundb = TESTDB_NAME + "_{}".format(i)

        conn = MongoClient('localhost', 27017)
        db = getattr(conn, rundb)
        collection = db.test_perovskites

        launchpad = LaunchPad(name=rundb)
        launchpad.reset(password=None, require_password=False)
        launchpad.add_wf(wf_creator(random_guess(dim), predictor, get_z, launchpad))

        y = []
        cands = 0
        # x = range(n_iterations)
        # for _ in range(n_iterations):
        while cands != n_cands:
            launch_rocket(launchpad)
            cands = collection.find({'yi':30.0}).count()
            y.append(cands)

        pickle.dump(y, open(filename + "_{}".format(i), 'w'))

        Y.append(y)
        launchpad.connection.drop_database(TESTDB_NAME)

    pickle.dump(Y, open(filename, 'w'))