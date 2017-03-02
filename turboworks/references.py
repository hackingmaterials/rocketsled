import numpy as np

class Dtypes(object):
    def __init__(self):
        d = np.sctypes
        self.ints = d['int'] + d['uint'] + [int]
        self.floats = d['float'] + [float]
        self.reals = self.ints + self.floats
        self.complex = d['complex']
        self.numbers = self.reals + self.complex
        self.others = d['others']
        self.all = self.numbers + self.others

dtypes = Dtypes()
