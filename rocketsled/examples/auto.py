"""
An example of the using rocketsled to auto_setup a loop using a custom function.

This script creates a runnable file which sets up an optimization loop as a 
self-repeating FireWorks workflow. 

Your auto-created script will be in the rocketsled/auto_sleds directory!
"""
from rocketsled import auto_setup


# The objective function must accept a vector and return a scalar.
def f(x):
    return x[0] * x[1] / x[2]


if __name__ == "__main__":
    dimensions = [(1, 100), (200, 300), (5.0, 10.0)]

    # Define the db where the LaunchPad and optimization data will be stored
    # The 'opt_label' field defines the name of the optimization collection
    dbinfo = {"host": "localhost", "name": "my_db", "port": 27017,
              "opt_label": "quickstart"}

    auto_setup(f, dimensions, wfname="quickstart", maximize=True, **dbinfo)
