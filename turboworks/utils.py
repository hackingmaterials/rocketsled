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

    Args:
        dimensions ([tuple]): defines the dimensions of each parameter
            example: [(1,50),(-18.939,22.435),["red", "green" , "blue", "orange"]]

    Returns:
        new_input (list): randomly chosen next parameters in the search space
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



def find_dupes(host='localhost', port=27017, opt_label='opt_default'):
    """
    For testing and development. Finds duplicate 'z' entries in the optdb.

    Args:
        host (string): host of the database
        port (int): port of the database

    Return:
        (list) of entries which are duplicates, including duplicates of duplicates
    """

    mongo = MongoClient(host=host, port=port)
    collection = getattr(mongo.turboworks, opt_label)

    dupes = []
    unique = []
    for doc in collection.find():
        if doc['z'] not in unique:
            unique.append(doc['z'])
        else:
            dupes.append(doc['z'])

    return dupes