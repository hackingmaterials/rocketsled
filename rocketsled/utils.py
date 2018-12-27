"""
Utility functions for OptTask.
"""

import imp
import random

import numpy as np

__author__ = "Alexander Dunn"
__email__ = "ardunn@lbl.gov"


class Dtypes(object):
    """
    Defines the datatypes available for optimization.
    """

    def __init__(self):
        d = np.sctypes
        self.ints = d['int'] + d['uint'] + [int]
        self.floats = d['float'] + [float]
        self.reals = self.ints + self.floats
        self.complex = d['complex']
        self.numbers = self.reals + self.complex
        self.others = d['others']
        self.bool = [bool, np.bool_]
        self.discrete = self.ints + self.others
        self.all = self.numbers + self.others


def deserialize(fun):
    """
    Takes a fireworks serialzed function handle and maps to a function object.

    Args:
        fun (string): a 'module.function' or '/path/to/mod.func' style string
            specifying the function

    Returns:
        (function) The function object defined by fun
    """
    toks = fun.rsplit(".", 1)
    modname, funcname = toks
    if "/" in toks[0]:
        modpath, modname = toks[0].rsplit("/", 1)
        mod = imp.load_source(modname, toks[0] + ".py")
    else:
        mod = __import__(str(modname), globals(), locals(),
                         fromlist=[str(funcname)])
    return getattr(mod, funcname)


def random_guess(dimensions, dtypes=Dtypes()):
    """
    Returns random new inputs based on the dimensions of the search space.
    It works with float, integer, and categorical types

    Args:
        dimensions ([tuple]): defines the dimensions of each parameter
            example: [(1,50),(-18.939,22.435),["red", "green" , "blue"]]

    Returns:
        random_vector (list): randomly chosen next params in the search space
            example: [12, 1.9383, "green"]
    """

    random_vector = []
    for dimset in dimensions:
        upper = dimset[1]
        lower = dimset[0]
        if type(lower) in dtypes.others or len(dimset) > 2:
            domain_size = len(dimset) - 1
            new_param = random.randint(0, domain_size)
            random_vector.append(dimset[new_param])
        elif type(lower) in dtypes.ints:
            new_param = random.randint(lower, upper)
            random_vector.append(new_param)
        elif type(lower) in dtypes.floats:
            new_param = random.uniform(lower, upper)
            random_vector.append(new_param)
        else:
            raise TypeError("The type {} is not supported by dummy opt as a "
                            "categorical or numerical type".format(type(upper)))
    return random_vector


def latex_float(f):
    """
    Convert floating point number into latex-formattable string for visualize.
    Might relocate to viz.py

    Args:
        f (float): A floating point number

    Returns:
        float_str (str): A latex-formatted string representing f.
    """
    float_str = "{0:.3g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def pareto(Y, maximize=False):
    """
    Returns the indices of Pareto-optimal solutions.

    Args:
        Y [list]: A list of lists containing values to be evaluated for Pareto-
            optimality

    Returns:
        list - The indices of the entries which are Pareto-optimal
    """
    Y = np.asarray(Y)
    po = np.ones(Y.shape[0], dtype=bool)
    for i, c in enumerate(Y):
        if maximize:
            po[i] = np.all(np.any(Y <= c, axis=1))
        else:
            po[i] = np.all(np.any(Y >= c, axis=1))
    return po


def convert_value_to_native(val, dtypes=Dtypes()):
    """
    Convert a single value to the native datatype for storage in the opt db.

    Args:
        val (int/float/str): Numpy or native implementation of numeric or
            categrical dtype
        dtypes (Dtypes): An instance of the Dtypes class

    Returns:
        native (int/float/str): The native python value of val.
    """
    if type(val) in dtypes.all:
        if type(val) in dtypes.floats:
            native = float(val)
        elif type(val) in dtypes.ints:
            native = int(val)
        elif type(val) in dtypes.bool:
            native = val
        elif type(val) in dtypes.others:
            native = str(val)
        else:
            TypeError("Dtype {} not found in rocketsled dtypes."
                      "".format(type(val)))
    else:
        TypeError("Dtype {} not found in rocketsled dtypes."
                  "".format(type(val)))
    return native


def split_xz(xz, x_dims, x_only=False):
    """
    Split concatenated xz vector into x and z vectors.

    Args:
        xz (list): The XZ matrix.
        x_dims ([list/tuple]) the dimensions of the X dimensions
        x_only (bool): If True, returns only the x vector.

    Returns:
        x, z (list, list): the separate X and Z matrices.

    """
    x, z = xz[:len(x_dims)], xz[len(x_dims):]
    if x_only:
        return x
    else:
        return x, z
