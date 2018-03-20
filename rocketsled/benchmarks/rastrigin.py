import math
import numpy as np
from rocketsled import auto_setup

def rastrigin(X):
    return 10*len(X) + sum([(x**2 - 10 * np.cos(2 * math.pi * x)) for x in X])

def rastrigindim (dim):
    return [(-5.12, 5.12)] * dim


if __name__ == "__main__":
    auto_setup(rastrigin, rastrigindim(6), wfname="rastrigin", launch_ready=True)
    