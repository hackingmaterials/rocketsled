"""
Utility functions for turboworks.
"""

from random import uniform, randint
from turboworks.references import dtypes
from pymongo import MongoClient

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"


def random_guess(dimensions):
    """
    Returns random new inputs based on the dimensions of the search space.
    It works with float, integer, and categorical types

    :param dimensions (list of 2-tuples and/or lists of strings): defines the dimensions of each parameter
        example:[ (1,50),(-18.939,22.435),["red", "green" , "blue", "orange"]]

    :return: new_input (list): randomly chosen next parameters in the search space
        example: [12, 1.9383, "green"]
    """

    new_input = []

    for dimset in dimensions:
        upper = dimset[1]
        lower = dimset[0]
        if type(lower) in dtypes.ints:
            new_param = randint(lower, upper)
            new_input.append(new_param)
        elif type(lower) in dtypes.floats:
            new_param = uniform(lower, upper)
            new_input.append(new_param)
        elif type(lower) in dtypes.others:
            domain_size = len(dimset)-1
            new_param = randint(0,domain_size)
            new_input.append(dimset[new_param])
        else:
            raise TypeError("The type {} is not supported by dummy opt as a categorical or "
                            "numerical type".format(type(upper)))


    return new_input



def find_dupes(host='localhost', port=27017, opt_label='Unnamed'):
    """
    For testing and development. Finds duplicate 'z' entries in the optdb.

    :param host: host of the optdb
    :param port: port of the optdb
    :return: (list) of entries which are duplicates, including duplicates of duplicates
    """

    mongo = MongoClient(host=host, port=port)
    db = mongo.turboworks.turboworks

    dupes = []
    unique = []
    for doc in db.find({'opt_label':opt_label}):
        if doc['z'] not in unique:
            unique.append(doc['z'])
        else:
            dupes.append(doc['z'])

    return dupes