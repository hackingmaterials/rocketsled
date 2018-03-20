from __future__ import unicode_literals

"""
Custom optimizers compatible with OptTask when used as predictors. Most 
algorithms here are specialized for specific kinds of optimization tasks. The 
optimization problem specifications necessary for each algorithm are given as 
documentation under each function.

Use these with OptTask by specifying the 'predictor' argument as the path
or pkg/module location of this file. For example:

predictor='rocketsled.opt.stochastic_rbf'

or 

predictor='my/path/to/opt.stochastic_rbf

Can't find one suiting your needs? Use these as a template for writing your own!
"""


def stochastic_rbf(X, y, dim, *args, **kwargs):
    return solution


def dycors_rbf(X, y, dim, *args, **kwargs):
    pass


def genetic_algorithm():
    pass


def particle_swarm():
    pass


def scatter_search():
    pass