"""
This module contains general references for the turboworks package.
"""

import numpy as np


__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"


# TODO: I don't see the reason to have two separate 'utility'-type files, i.e. references.py and utils.py
# Combine this and the utility function for random guesses into a single file. Or move this into optimize.py.

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

