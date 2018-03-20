from  __future__ import print_function, unicode_literals

"""
Utility functions for OptTask.
"""

import imp
import random
from numpy import sctypes

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"


class Dtypes(object):
    """
    Defines the datatypes available for optimization.
    """

    def __init__(self):
        d = sctypes
        self.ints = d['int'] + d['uint'] + [int]
        self.floats = d['float'] + [float]
        self.reals = self.ints + self.floats
        self.complex = d['complex']
        self.numbers = self.reals + self.complex
        self.others = d['others']
        self.discrete = self.ints + self.others
        self.all = self.numbers + self.others

def deserialize(fun):
    """
    Takes a fireworks serialzed function handle and maps it to a function object.

    Args:
        fun (string): a 'module.function' or '/path/to/mod.func' style string specifying the function

    Returns:
        (function) The function object defined by fun
    """
    # todo: merge with PyTask's deserialize code, move to fw utils

    toks = fun.rsplit(".", 1)
    modname, funcname = toks
    if "/" in toks[0]:
        modpath, modname = toks[0].rsplit("/", 1)
        packages = imp.load_source(modname, toks[0] + ".py")

    mod = __import__(str(modname), globals(), locals(), fromlist=[str(funcname)])
    return getattr(mod, funcname)


def random_guess(dimensions, dtypes=Dtypes()):
    """
    Returns random new inputs based on the dimensions of the search space.
    It works with float, integer, and categorical types

    Args:
        dimensions ([tuple]): defines the dimensions of each parameter
            example: [(1,50),(-18.939,22.435),["red", "green" , "blue", "orange"]]

    Returns:
        random_vector (list): randomly chosen next parameters in the search space
            example: [12, 1.9383, "green"]
    """

    random_vector = []
    for dimset in dimensions:
        upper = dimset[1]
        lower = dimset[0]
        if type(lower) in dtypes.ints:
            new_param = random.randint(lower, upper)
            random_vector.append(new_param)
        elif type(lower) in dtypes.floats:
            new_param = random.uniform(lower, upper)
            random_vector.append(new_param)
        elif type(lower) in dtypes.others:
            domain_size = len(dimset)-1
            new_param = random.randint(0, domain_size)
            random_vector.append(dimset[new_param])
        else:
            raise TypeError("The type {} is not supported by dummy opt as a "
                            "categorical or numerical type".format(type(upper)))
    return random_vector