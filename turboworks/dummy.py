from random import uniform, randint
import numpy as np

def dummy_minimize(dimensions):
    """
    This function returns random new inputs based on the dimensions of the search space.
    It works with float, integer, and categorical types

    :param dimensions (list of 2-tuples and/or lists of strings): defines the dimensions of each parameter
        example:[ (1,50),(-18.939,22.435),["red", "green" , "blue", "orange"]]

    :return: new_input (list): randomly chosen next parameters in the search space
        example: [12, 1.9383, "green"]
    """

    try:
        basestring
    except NameError:  # Python3 compatibility
        basestring = str

    new_input = []

    for dimset in dimensions:
        upper = dimset[1]
        lower = dimset[0]
        if type(upper) == np.int64 or type(upper) == int:
            new_param = randint(lower, upper)
            new_input.append(new_param)
        elif type(upper) == np.float64 or type(upper) == float:
            new_param = uniform(lower, upper)
            new_input.append(new_param)
        elif isinstance(upper, basestring) or isinstance(upper, unicode) or isinstance(upper, np.unicode_):
            domain_size = len(dimset)-1
            new_param = randint(0,domain_size)
            new_input.append(dimset[new_param])
        else:
            raise TypeError("The type {} is not supported by dummy opt as a categorical or "
                            "numerical type".format(type(upper)))


    return new_input