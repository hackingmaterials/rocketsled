from __future__ import unicode_literals

"""
Acquisition functions and utilities.
"""
from multiprocessing import Pool, cpu_count


def acquire(acq, X, Y, space, model, maximize):
    """
    A high level function for calculating acquisition values.

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

    Returns:
        (list) acquisition values for space. Higher acquisition values
            correspond to better guesses.
    """

    if model.__name__ == "GaussianProcessRegressor":
        model.fit(X, Y)
        mu, std = model.predict(space, return_std=True)
    else:
        mu, std = bootstrap(X, Y, space, model)

    if acq == 'ei':
        acq = ei
    elif acq == 'pi':
        acq = pi
    elif acq == 'lcb':
        acq = lcb
    else:
        raise ValueError("Unknown acquisition function!")

    return [acq(mu[i], std[i], maximize) for i in range(len(mu))]

def bootstrap(X, Y, space, model):

    # 1. create random subsets of indices for training

    # 2. Train N models on these subsets in parallel

    # 3. Predict over the entire space

    # 4. Take the mean and std of the space

    mu = 1
    std = 0
    return (mu, std)

# Acquisition methods should return the acquisition values
def ei(mu, std, maximize):
    pass

def pi(mu, std, maximize):
    pass

def lcb(mu, std, maximize):
    pass


if __name__ == "__main__":
    pass