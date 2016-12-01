"""
This file contains utility functions for discrete space calculations.
"""

import itertools
import numpy as np
from random import randint
import warnings
import resource


def calculate_discrete_space(dimensions):
    """
    This is a utility function for calculating all entries in a discrete space.
    For example, if your dimensions are

     [(1,2), ["red","blue"]]

    calculate_discrete_space will return all possible combinations of these dimensions' entries:

    [(1, 'red'), (1, 'blue'), (2, 'red'), (2, 'blue')]

    In duplicate checking for discrete spaces, the generated list will be narrowed down until no entries remain.

    WARNING: Very large discrete spaces will cause a memory bomb. Typically a space of about 1,000 entries takes
    0.005s to compute, but larger spaces can take much longer (or may just nuke your computer RAM, be careful).

    """
    total_dimspace = []

    for dimension in dimensions:
        if type(dimension[0]) == int or type(dimension[0]) == np.int64:
            # Then the dimension is of the form (lower, upper)
            lower = dimension[0]
            upper = dimension[1]
            dimspace = list(range(lower, upper + 1))
        elif type(dimension[0]) == float or type(dimension[0]) == np.float64:
            # The chance of a random sample of identical float is nil
            raise ValueError("The dimension is a float. The dimension space is infinite.")
        else:  # The dimension is a discrete finite string list
            dimspace = dimension
        total_dimspace.append(dimspace)

    if len(dimensions)==1:
        return [[x] for x in total_dimspace[0]]
    else:
        return list(itertools.product(*total_dimspace))


def duplicate_check(new_input, opt_inputs, X, optimizer):
    """
    This is a utility function for checking duplicates and returning new discrete entries until the search space is
    exhausted.

    new_input (list or tuple): the input to be checked agains the space
    opt_inputs (list of lists of list of tuples): the list of previous inputs
    X (list of lists or list of tuples): comprehensive list of all entries in the space (use calculate_discrete_space)

    """
    warnings.simplefilter('always', UserWarning)
    if list(new_input) in opt_inputs or tuple(new_input) in opt_inputs:
        remaining_inputs = X
        for element in opt_inputs:
            while element in remaining_inputs:
                remaining_inputs.remove(element)
            while tuple(element) in remaining_inputs:
                remaining_inputs.remove(tuple(element))

        if len(remaining_inputs) == 0:
            warnings.warn('All search combinations in the space have been exhausted. '
                          'Repeating calculation based on {} optimization.'.format(optimizer))
            return new_input
        else:
            index = randint(0, len(remaining_inputs) - 1)
            return list(remaining_inputs[index])
    else:
        return new_input