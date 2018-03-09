from __future__ import print_function, unicode_literals

"""
An example of the using rocketsled to auto_setup a loop using a custom function.
"""
from rocketsled import auto_setup

def myfun(x):
    res = float(x[0]) * float(x[1]) / float(x[2])
    return res


if __name__ == "__main__":
    auto_setup(myfun, dimensions=[(1, 100), (1, 100), (5, 65)],
               wfname="expensive_WF", host='localhost',
               name="ROCKETSLED_EXAMPLES", port=27017, opt_label='opt_auto',
               max=True)