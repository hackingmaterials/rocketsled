from __future__ import print_function, unicode_literals

"""
An example of the using rocketsled to auto_setup a loop using a custom function.
"""
from rocketsled import auto_setup

def myfun(x):
    res = (float(x[0]) + float(x[1]) - float(x[2]))
    return res


if __name__ == "__main__":
    auto_setup(myfun, dimensions=[(1, 10), (1, 10), (2, 30)],
               wfname="my_test_wf", host='localhost',
               name="ROCKETSLED_EXAMPLES", port=27017, opt_label='opt_auto')