from __future__ import unicode_literals, print_function, division

"""
Automatically set up a file to execute a rocketsled optimization given:
    - a workflow taking x and returning y
    - (optionally) an optimizer
    
All output will be organized inside the "auto_sleds" directory. 
"""
import os
import datetime
from fireworks import PyTask, Firework, Workflow
from fireworks.utilities.fw_utilities import FW_BLOCK_FORMAT
from rocketsled import OptTask
from rocketsled.utils import deserialize


def auto_setup(func, name=None, opttask_kwargs=None):
    """
    Automatically set up a FireWorks-based optimization loop with OptTask.
    Use either a function, Workflow, or Firework to pass into auto_setup.

    Args:
         fname (str): The full path of the python script which will be written.
         func (str, function object):
         wf (Firework/Workflow):
    """

    # Determine the name and directory
    dir = os.path.dirname(os.path.abspath(__file__)) + '/auto_sleds'
    if not os.path.exists(dir):
        os.mkdir(dir)
    if name:
        if "/" in name:
            raise ValueError("Please do not use the '/' character in the name.")
    else:
        time_now = datetime.datetime.utcnow().strftime(FW_BLOCK_FORMAT)
        name = "autosled_" + time_now
    filename = dir + "/" + name

    wfc = "rocketsled.auto_sleds." + name + ".workflow_creator"

    if func is not None:
        if not isinstance(func, str):
            if not hasattr(func, '__call__'):
                raise TypeError("func must be a callable function.")

    else:
        raise ValueError("Please specify a callable function or a properly"
                         "formatted string location of the function")


def OptTask_as_string(opttask_kwargs):
    otstr = "OptTask("
    for k, v in opttask_kwargs.items():
        if isinstance(v, (str, unicode)):
            strv = "'{}'".format(str(v))
        else:
            strv = str(v)
        otstr += str(k) + "=" + strv + ", "
    otstr = otstr[:-2] + ")"
    return otstr

if __name__ == "__main__":
    print(OptTask_as_string({'dimensions': [(1, 100), (2, 20)], 'lpad': 'my_launchpad.yaml'}))