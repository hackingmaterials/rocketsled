import numpy as np
import pandas as pd
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from matminer.data_retrieval.retrieve_MongoDB import MongoDataRetrieval
# from matminer.descriptors.composition_features import get_pymatgen_descriptor
from pymongo import MongoClient
from references import Evaluator

pd.set_option('display.width', 2000)
pd.set_option('display.max_colwidth', 500)
pd.set_option('max_columns', None)
pd.set_option('display.max_rows', None)

api_key = 'AJsTCV3n1IOkBi97'
mp_retriever = MPDataRetrieval(api_key)


def pretty_formula(i, data):
    A = data.A[i]
    B = data.B[i]
    anion = data.anion[i]
    return A+B+anion

def score_n_store():

    fit_eval = Evaluator()

    client = MongoClient('localhost', 27017)
    unc = client.unc.data_raw

    mdb_retriever = MongoDataRetrieval(unc)
    scoring_features = ['A', 'B', 'anion', 'gllbsc_dir-gap', 'gllbsc_ind-gap', 'heat_of_formation_all', 'VB_dir',
                        'CB_dir',
                        'VB_ind', 'CB_ind']
    data = mdb_retriever.get_dataframe(scoring_features)


    score_array = []

    for i in range(18928):

        if i%1000==0:
            print(i)


        vb_ind = data.VB_ind[i]
        cb_ind = data.CB_ind[i]
        vb_dir = data.VB_dir[i]
        cb_dir = data.CB_dir[i]
        gap_ind = data['gllbsc_ind-gap'][i]
        gap_dir = data['gllbsc_dir-gap'][i]
        heat_of_formation = data.heat_of_formation_all[i]

        s = fit_eval.simple(gap_dir, gap_ind, heat_of_formation, vb_dir, cb_dir, vb_ind, cb_ind)
        c = fit_eval.complex(gap_dir, gap_ind, heat_of_formation, vb_dir, cb_dir, vb_ind, cb_ind)
        p = fit_eval.complex_product(gap_dir, gap_ind, heat_of_formation, vb_dir, cb_dir, vb_ind, cb_ind)

        score_row = [s,c,p]
        score_array.append(score_row)


    scores = pd.DataFrame(np.array(score_array), columns=['simple_score', 'complex_score', 'product_score'])
    data = pd.concat([data,scores], axis=1)

    data.to_csv('unc.csv')

    # proof of accuracy (compare with paper found candidates)
    hits = data[data.complex_score.isin([30, 30.0])]
    print(hits)

def add_mp_attribs():
    data = pd.read_csv('unc.csv')
    for i in range(1):
        pform = pretty_formula(i, data)
        mp_data = mp_retriever.get_dataframe(criteria={"pretty_formula":"LiVO3"},
                                             properties= ["spacegroup"])
        print(mp_data)


if __name__=="__main__":
    add_mp_attribs()


    # mp_data = mp_retriever.get_dataframe()
