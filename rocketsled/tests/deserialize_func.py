"""
Auxiliary file for testing utils.
"""


def obj_func(x):
    """
    Objective function which sums the elements in x.

    Args:
        x ([float] or [int]): List of numbers to sum.

    Returns:
        y (float or int): The sum of x.

    """
    return sum(x)
