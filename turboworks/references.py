"""
This module contains general references for the turboworks package.
"""

import numpy as np


__author__ = "Alexander Dunn"
__version__ = "0.1"
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
        self.discrete = self.ints + self.others
        self.all = self.numbers + self.others

dtypes = Dtypes()

example_data = [{'A': 1, 'B': 2, 'C': 1, 'D': 'blue', 'output': 19.121},
                 {'A': 2, 'B': 2, 'C': 2, 'D': 'red', 'output': 81.2},
                 {'A': 1, 'B': 1, 'C': 1, 'D': 'blue', 'output': 15.3}]
