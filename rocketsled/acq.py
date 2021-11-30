"""
Acquisition functions and utilities.
"""

from copy import deepcopy
from multiprocessing import cpu_count

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.model_selection import train_test_split

__author__ = "Alexander Dunn"
__email__ = "ardunn@lbl.gov"


# def acquire(acq, X, Y, space, model, nstraps, return_means=False):
#     """
#     A high level function for calculating acquisition values. Includes a
#     strategy for estimating mean values and uncertainty with bootstrapping;
#     Independently train with different sets of data, and predict over the same
#     space of unknown points. Assumes minimization!
#
#     Args:
#         acq (str): The acquisition function ('ei', 'pi', or 'lcb')
#         X ([list]): A list of x vectors (inputs), for training.
#         Y (list): A list of scalars (outputs), for training.
#         space ([list]): A list of possible X vectors, yet to be explored. This
#             is the 'test' set.
#         model (BaseEstimator object): sklearn estimator object. Must have .fit
#             and .predict methods.
#         nstraps (int): The number of bootstrap samplings, with replacement,
#             which will be performed. This is also the number of regressor
#             fittings and predictions which will be performed.
#         return_means (bool): If True, also return the mean of the acquisition
#             function alongside the acquisition fucntion values.
#
#     Returns:
#         (list) acquisition values for space. Higher acquisition values
#             correspond to better guesses.
#     """
#
#     if model.__class__.__name__ == "GaussianProcessRegressor":
#         model.fit(X, Y)
#         mu, std = model.predict(space, return_std=True)
#     else:
#         predicted = Parallel(n_jobs=cpu_count())(
#             delayed(ppredict)(X, Y, space, model) for _ in np.zeros(nstraps)
#         )
#         mu = np.mean(predicted, axis=0)
#         std = np.std(predicted, axis=0)
#
#     if acq == "ei":
#         acqf = ei
#     elif acq == "pi":
#         acqf = pi
#     elif acq == "lcb":
#         acqf = lcb
#     else:
#         raise ValueError("Unknown acquisition function: {}!".format(acq))
#
#     if return_means:
#         return acqf(min(Y), mu, std).tolist(), mu
#     else:
#         return acqf(min(Y), mu, std).tolist()


def acquire(acq, Y, mu, std, return_means=False):
    if acq == "ei":
        acqf = ei
    elif acq == "pi":
        acqf = pi
    elif acq == "lcb":
        acqf = lcb
    else:
        raise ValueError("Unknown acquisition function: {}!".format(acq))

    if return_means:
        return acqf(min(Y), mu, std).tolist(), mu
    else:
        return acqf(min(Y), mu, std).tolist()


def predict(X, Y, space, model, nstraps):
    if model.__class__.__name__ == "GaussianProcessRegressor":
        model.fit(X, Y)
        mu, std = model.predict(space, return_std=True)
    else:
        predicted = Parallel(n_jobs=cpu_count())(
            delayed(ppredict)(X, Y, space, model) for _ in np.zeros(nstraps)
        )
        mu = np.mean(predicted, axis=0)
        std = np.std(predicted, axis=0)

    return mu, std


def ppredict(X, Y, space, model):
    """
    Run a split and fit on a random subsample of the entire explored X. Use this
    fitted model to predict the remaining space. Meant to be run in parallel in
    combination with joblib's delayed and Parallel utilities.

    Args:
        X ([list]): A list of x vectors, for training.
        Y (list): A list of scalars, for training.
        space ([list]): A list of possible X vectors, yet to be explored. This
            is the 'test' set.
        model (BaseEstimator object): sklearn estimator object. Must have .fit
            and .predict methods.

    Returns:
        (numpy array): The 1-D array of predicted points for the entire
            remaining space.

    """
    X_train, _, y_train, _ = train_test_split(X, Y, test_size=0.25)
    pmodel = deepcopy(model)
    pmodel.fit(X_train, y_train)
    return pmodel.predict(space)


def ei(fmin, mu, std, xi=0.01):
    """
    Returns expected improvement values.

    Args:
        fmin (float): Minimum value of the objective function known thus far.
        mu (numpy array):  Mean value of bootstrapped predictions for each y.
        std (numpy array): Standard deviation of bootstrapped predictions for
            each y.
        xi (float): Amount of expected improvement, optional hyper-parameter.
            Default value taken from "Practical bayesian optimization" by Daniel
            Lizotte (2008).

    Returns:
        vals (numpy array): Acquisition values.

    """
    vals = np.zeros_like(mu)
    mask = std > 0
    stdm = std[mask]
    improve = fmin - mu[mask] - xi
    vals[mask] = improve * norm.cdf(improve / stdm) + stdm * norm.pdf(improve / stdm)
    # improve = fmin - mu
    # vals = improve * norm.cdf(improve/std) + std * norm.pdf(improve/std)
    return vals


def pi(fmin, mu, std, xi=0.01):
    """
    Returns probability of improvement values.

    Args:
        fmin (float): Minimum value of the objective function known thus far.
        mu (numpy array):  Mean value of bootstrapped predictions for each y.
        std (numpy array): Standard deviation of bootstrapped predictions for
            each y.
        xi (float): Amount of expected improvement, optional hyper-parameter.
            Default value taken from "Practical bayesian optimization" by Daniel
            Lizotte (2008).

    Returns:
        vals (numpy array): Acquisition values.

    """
    vals = np.zeros_like(mu)
    mask = std > 0
    vals[mask] = norm.cdf((fmin - mu[mask] - xi) / std[mask])
    return vals


def lcb(fmin, mu, std, kappa=1.96):
    """
    Returns lower confidence bound estimates.
        fmin (float): (not used): Minimum value of the objective function known
            thus far.
        mu (numpy array):  Mean value of bootstrapped predictions for each y.
        std (numpy array): Standard deviation of bootstrapped predictions for
            each y.
        kappa (float): Controls the variance in the prediction,
            affecting exploration/exploitation.

    Returns:
        vals (numpy array): Acquisition values.
    """
    return mu - kappa * std
