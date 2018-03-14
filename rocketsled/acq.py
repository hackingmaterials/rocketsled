from __future__ import unicode_literals

"""
Acquisition functions and utilities.
"""
from copy import deepcopy
from multiprocessing import cpu_count
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.externals.joblib import Parallel, delayed

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"

def acquire(acq, X, Y, space, model, maximize, nstraps):
    """
    A high level function for calculating acquisition values. Includes a
    strategy for stimating mean values and uncertainty with bootstrapping;
    Independently train with different sets of data, and predict over the same
    space of unknown points.

    Args:
        acq (str): The acquisition function ('ei', 'pi', or 'lcb')
        X ([list[): A list of x vectors, for training.
        Y (list): A list of scalars, for training.
        space ([list[): A list of possible X vectors, yet to be explored. This
            is the 'test' set.
        model (BaseEstimator object): sklearn estimator object. Must have .fit
            and .predict methods.
        maximize (bool): Whether to look for points to maximize or points to
            minimize.
        nstraps (int): The number of bootstrap samplings, with replacement,
            which will be performed. This is also the number of regressor
            fittings and predictions which will be performed.

    Returns:
        (list) acquisition values for space. Higher acquisition values
            correspond to better guesses.
    """

    if model.__class__.__name__ == "GaussianProcessRegressor":
        model.fit(X, Y)
        mu, std = model.predict(space, return_std=True)
    else:
        predicted = Parallel(n_jobs=cpu_count())(
            delayed(ppredict)(X, Y, space, model) for _ in np.zeros(nstraps))
        mu = np.mean(predicted, axis=0)
        std = np.std(predicted, axis=0)

    if acq == 'ei':
        acqf = ei
    elif acq == 'pi':
        acqf = pi
    elif acq == 'lcb':
        acqf = lcb
    else:
        raise ValueError("Unknown acquisition function!")

    if maximize:
        Y = -1 * np.asarray(Y)
        mu = -1 * mu

    return acqf(min(Y), mu, std).tolist()

def ppredict(X, Y, space, model):
    """
    Run a split and fit on a random subsample of the entire explored X. Use this
    fitted model to predict the remaining space. Meant to be run in parallel in
    combination with joblib's delayed and Parallel utilities.

    Args:
        X ([list[): A list of x vectors, for training.
        Y (list): A list of scalars, for training.
        space ([list[): A list of possible X vectors, yet to be explored. This
            is the 'test' set.
        model (BaseEstimator object): sklearn estimator object. Must have .fit
            and .predict methods.

    Returns:
        (numpy array): The 1-D array of predicted points for the entire
            remaining space.

    """
    X_train, _, y_train, _ = train_test_split(X, Y, test_size=0.2)
    pmodel = deepcopy(model)
    pmodel.fit(X_train, y_train)
    return pmodel.predict(space)


def ei(fmin, mu, std):
    """
    Returns expected improvement values.

    Args:
        fmin (float): Minimum value of the objective function known thus far.
        mu (numpy array):  Mean value of bootstrapped predictions for each y.
        std (numpy array): Standard deviation of bootstrapped predictions for
            each y.

    Returns:
        vals (numpy array): Acquisition values.

    """
    vals = np.zeros_like(mu)
    mask = std > 0
    improve = fmin - mu[mask]
    vals[mask] = improve * norm.cdf(improve/std) + std * norm.pdf(improve/std)
    return vals

def pi(fmin, mu, std):
    """
    Returns expected improvement values.

    Args:
        fmin (float): Minimum value of the objective function known thus far.
        mu (numpy array):  Mean value of bootstrapped predictions for each y.
        std (numpy array): Standard deviation of bootstrapped predictions for
            each y.

    Returns:
        vals (numpy array): Acquisition values.

        """
    vals = np.zeros_like(mu)
    mask = std > 0
    vals[mask] = norm.cdf((fmin - mu[mask])/std[mask])
    return vals

def lcb(fmin, mu, std):
    """
    Returns expected improvement values.

    Args:
        fmin (float): Minimum value of the objective function known thus far.
        mu (numpy array):  Mean value of bootstrapped predictions for each y.
        std (numpy array): Standard deviation of bootstrapped predictions for
            each y.

    Returns:
        vals (numpy array): Acquisition values.

        """
    # todo: this is broken. :(
    beta = 1.96
    return mu - beta * std

if __name__ == "__main__":
    pass