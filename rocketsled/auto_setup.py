from __future__ import unicode_literals, print_function, division

"""
Automatically set up a file to execute a rocketsled optimization given:
    - a workflow taking x and returning y
    - (optionally) an optimizer
"""

from fireworks import PyTask, Firework, Workflow


def auto_setup(fname, func=None, wf=None, opttask_args=None):
    """
    Automatically set up a FireWorks-based optimization loop with OptTask.
    Use either a function, Workflow, or Firework to pass into auto_setup.

    Args:
         fname (str): The full path of the python script which will be written.
         func (str, function object):
         wf (Firework/Workflow):
    """

    #todo: set the default opttask firework name to "RocketsledFW
    if wf is not None:
        if isinstance(wf, Firework):
            # create a workflow consisting of a firework and a opttask firework
            pass
        elif isinstance(wf, Workflow):
            # append a opttask firework to the workflow
            pass
        else:
            raise TypeError("The workflow must be a firework or a workflow.")

    if func is not None:
        # or maybe it could be string to pass to pytask??
        if not hasattr(func, '__call__'):
            raise TypeError("func must be a callable function.")
        # Make a single PyTask firework, then append a opttask firework.

    else:
        raise ValueError("Please specify either a function or a workflow.")


