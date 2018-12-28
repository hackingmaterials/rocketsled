"""
Utility functions for OptTask.
"""
import os
import imp
import random
from collections.abc import Iterable

import numpy as np
from ruamel.yaml import YAML

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


dtypes = Dtypes()


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


def get_default_opttask_kwargs():
    """
    Get the default configuration kwargs for OptTask.
    Returns:
        conf_dict (dict): The default kwargs for OptTask

    """
    cwd = os.path.dirname(os.path.realpath(__file__))
    fname = os.path.join(cwd, "defaults.yaml")
    with open(fname, 'r') as config_raw:
        yaml = YAML()
        conf_dict = dict(yaml.load(config_raw))
    return conf_dict


def check_dims(dims):
    """
    Ensure the dimensions are in the correct format for the optimization.

    Dimensions should be a list or tuple of lists or tuples each defining
    the search space in one dimension. The datatypes used inside each
    dimension's  definition should be NumPy compatible datatypes.

    Continuous numerical dimensions (floats and ranges of ints) should be
    2-tuples in the form (upper, lower). Categorical dimensions or
    discontinuous numerical dimensions should be exhaustive lists/tuples
    such as ['red', 'green', 'blue'] or [1.2, 11.5, 15.0].

    Args:
        dims (list): The dimensions of the search space.

    Returns:
        [str]: Types of the dimensions in the search space defined by dims.
    """
    dims_types = [list, tuple]
    dim_spec = []

    if type(dims) not in dims_types:
        raise TypeError("The dimensions must be a list or tuple.")

    for dim in dims:
        if type(dim) not in dims_types:
            raise TypeError("The dimension {} must be a list or tuple."
                            "".format(dim))

        for entry in dim:
            if type(entry) not in dtypes.all:
                raise TypeError("The entry {} in dimension {} cannot be "
                                "used with OptTask. A list of acceptable "
                                "datatypes is {}".format(entry, dim,
                                                         dtypes.all))
            for dset in [dtypes.ints,
                         dtypes.floats,
                         dtypes.others]:
                if type(entry) not in dset and type(dim[0]) in dset:
                    raise TypeError(
                        "The dimension {} contains heterogeneous"
                        " types: {} and {}".format(dim,
                                                   type(dim[0]),
                                                   type(entry)))
        if isinstance(dim, list):
            if type(dim[0]) in dtypes.ints:
                dim_spec.append("int_set")
            elif type(dim[0]) in dtypes.floats:
                dim_spec.append("float_set")
            elif type(dim[0]) in dtypes.others:
                dim_spec.append("categorical {}".format(len(dim)))
        elif isinstance(dim, tuple):
            if type(dim[0]) in dtypes.ints:
                dim_spec.append("int_range")
            elif type(dim[0]) in dtypes.floats:
                dim_spec.append("float_range")
            elif type(dim[0]) in dtypes.others:
                dim_spec.append("categorical {}".format(len(dim)))
    return dim_spec


def is_discrete(dims, criteria='all'):
    """
    Checks if the search space is discrete.

    Args:
        dims ([tuple]): dimensions of the search space
        criteria (str/unicode): If 'all', returns bool based on whether
            ALL dimensions are discrete. If 'any', returns bool based on
            whether ANY dimensions are discrete.

    Returns:
        (bool) whether the search space is totally discrete.
    """
    if criteria == 'all':
        for dim in dims:
            if type(dim[0]) not in dtypes.discrete or \
                    type(dim[1]) not in dtypes.discrete:
                return False
        return True
    elif criteria == 'any':
        for dim in dims:
            if type(dim[0]) in dtypes.discrete or \
                    type(dim[1]) in dtypes.discrete:
                return True
        return False


def convert_native(a):
    """
    Convert iterables of non-native types to native types for bson storage
    in the database. For situations where .tolist() does not work.

    Args:
        a (iterable or scalar): Input list of strings, ints, or
            floats, as either numpy or native types (or others), which
            will be force-coerced to native types. Also works with scalar
            entries such as floats, ints, etc.

    Returns:
        native (list): A list of the data in a, converted to native types.

    """
    if isinstance(a, Iterable):
        try:
            # numpy conversion
            native = a.tolist()
        except AttributeError:
            native = [None] * len(a)
            for i, val in enumerate(a):
                try:
                    native[i] = val.item()
                except AttributeError:
                    native[i] = convert_value_to_native(val, dtypes)
    else:
        native = convert_value_to_native(a, dtypes)
    return native


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


