import itertools
import numpy as np

"""
This file holds a utility function for calculating all entries in a discrete space.
For example, if your dimensions are

 [(1,2), ["red","blue"]]

calculate_discrete_space will return all possible combinations of these dimensions' entries:

[(1, 'red'), (1, 'blue'), (2, 'red'), (2, 'blue')]

In duplicate checking for discrete spaces, the generated list will be narrowed down until no entries remain.

WARNING: Very large discrete spaces will memory hog this process. Typically a space of about 1,000 entries takes
0.005s to compute, but larger spaces can take much longer.

"""

def calculate_discrete_space(dimensions):
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
