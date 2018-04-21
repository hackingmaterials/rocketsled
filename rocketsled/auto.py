from __future__ import unicode_literals, print_function, division

"""
Automatically set up a file to execute a rocketsled optimization given:
    - a function taking in a vector x, and returning a scalar y (func)
    - a space you'd like to constrain the problem to (dimensions)
    
    Thats it!
    
All output will be organized inside the "auto_sleds" directory. 
"""

import os
import sys
import datetime
import warnings
from fireworks import LaunchPad
from fireworks.utilities.fw_utilities import FW_BLOCK_FORMAT
from rocketsled.utils import deserialize


def auto_setup(func, dimensions, wfname=None, launch_ready=False, **kwargs):
    """
    Automatically set up a FireWorks-based optimization loop with OptTask with
    you own function.

    The loop is set up as a script, which is written to the
    rocketsled/auto_sleds directory. All you need to do it set up your
    Fireworks launchpad and run the script created, to get started.

    Make sure to pass in necessary launchpad data to OptTask through kwargs of
    this function!

    Args:

        func (function object): A function object accepting a single positional
            argument, x, a vector of ints/floats/strs and returning a single
            scalar, y.
        wfname (str): The base name you want for the workflow.
        dimensions (list): A list of dimensions constraining each of the
            variables in x. Each 2-tuple in the list defines one dimension in
            the search space in (low, high) format. For categorical dimensions,
            includes all possible categories as a list.
            Example:
            dimensions = [(1,100), (9.293, 18.2838), ("red", "blue", "green")]
        launch_ready (bool): If True, the created document can be executed
            immediately.
        kwargs: Arguments to be passed as options to OptTask. Valid arguments
            to be passed are any valid args for OptTask. For example,
            lpad, host, port, name, opt_label, acq, predictor, etc...

    """
    # Determine the name and directory
    dir = os.path.dirname(os.path.abspath(__file__)) + '/auto_sleds'
    time_now = datetime.datetime.utcnow().strftime(FW_BLOCK_FORMAT)

    if not os.path.exists(dir):
        os.mkdir(dir)
    if not os.path.exists(dir + "/__init__.py"):
        with open(dir + "/__init__.py", "w") as ipy:
            ipy.write('"""\n This file has been autocreated by '
                      'auto_setup.py\n"""')
    if wfname:
        if "/" in wfname or " " in wfname:
            raise ValueError("Please do not use ' ' or '/' in the wf name.")
    else:
        wfname = "autosled_" + time_now

    filename = dir + "/" + wfname + ".py"
    if os.path.exists(filename):
        warnings.warn("That workflow file has already been created! Appending "
                      "the current datetime to the filename.")
        filename = dir + "/" + wfname + "_" + time_now + ".py"
        wfname += "_" + time_now

    wfc = "rocketsled.auto_sleds." + wfname + ".wf_creator"
    kwargs['wf_creator'] = wfc
    kwargs['dimensions'] = dimensions

    if func is not None:
        if not hasattr(func, '__call__'):
            raise TypeError("func must be a callable function.")
        rawfunc = os.path.abspath(sys.modules.get(func.__module__).__file__)
        if rawfunc.endswith(".pyc"):
            rawfunc = rawfunc[:-3] + func.__name__
        else:
            rawfunc = rawfunc[:-2] + func.__name__

        # See if import will work
        try:
            deserialize(rawfunc)
        except AttributeError:
            warnings.warn("Import attempt failed! File will still be written, "
                          "but if launching does not work, make sure the "
                          "function exists, is named properly, and "
                          "the string argument of the functions location is "
                          "/the/full/path/to/my/module.myfunc")

        funcpath = "rocketsled.auto_sleds." + wfname + ".f"

        with open(filename, 'w') as f:
            try:
                f.write("from __future__ import unicode_literals\n")
                f.write('"""\n')
                f.write("This is an automatically created script from auto_setup.\n"
                        "If you are not comfortable working with FireWorks and "
                        "PyTask, do NOT move this\nfile out this directory or "
                        "rename it if you want to run this workflow!\n\nIf you are"
                        " comfortable working with FireWorks and PyTask, feel f"
                        "ree to edit\nand/or move this file to suit your needs. "
                        "See the OptTask documentation and the\nexamples for more "
                        "information on setting up workflow creators.\n")
                f.write('"""\n')
                f.write("from fireworks import PyTask, Firework, Workflow, "
                        "LaunchPad\n")
                f.write("from fireworks.core.rocket_launcher import rapidfire\n")
                f.write("from rocketsled.utils import deserialize, "
                        "random_guess\n")
                f.write("from rocketsled import OptTask\n\n\n")
                f.write("# This is your function, imported to rocketsled to use"
                        " with PyTask.\n")
                f.write("f = deserialize('" + rawfunc + "')\n\n")
                f.write("def wf_creator(x):\n")
                f.write("    spec = {'_x_opt':x}\n")
                f.write("    pt = " + PyTask_as_string(funcpath) + "\n")
                f.write("    ot = " + OptTask_as_string(**kwargs) + "\n")
                f.write("    fw0 = Firework([pt], spec=spec, name='PyTaskFW')\n")
                f.write("    fw1 = Firework([ot], spec=spec, "
                        "name='RocketsledFW')\n")
                f.write("    wf = Workflow([fw0, fw1], {fw0: [fw1], fw1: []},"
                        " name='" + wfname + " @ ' + str(x))\n")
                f.write("    return wf\n")
                f.write("\n\nif __name__=='__main__': \n\n")
                f.write("    # Make sure the launchpad below is correct, and "
                        "make changes if necessary if\n    # it does not match "
                        "the OptTask db ^^^:\n")
                if all(s in kwargs for s in ['host', 'port', 'name']):
                    h = kwargs['host']
                    p = kwargs['port']
                    n = kwargs['name']
                    f.write("    lpad = LaunchPad(host='{}', port={}, "
                            "name='{}')\n".format(h, p, n))
                elif 'lpad' in kwargs:
                    if isinstance(kwargs['lpad'], LaunchPad):
                        lpad = kwargs['lpad'].to_dict()
                    else:
                        lpad = kwargs['lpad']
                    f.write("    lpad = LaunchPad.from_dict(" + lpad + ")\n")
                else:
                    f.write("    lpad = LaunchPad.auto_load()\n")
                f.write("    # lpad.reset(password=None, require_password=False"
                        ")\n")
                f.write("\n    # Define your workflow to start...\n")
                f.write("    wf1 = wf_creator(random_guess(" + str(dimensions) +
                        "))\n\n")
                f.write("    # Add it to the launchpad and launch!\n")
                f.write("    lpad.add_wf(wf1)\n")
                if launch_ready:
                    f.write("    rapidfire(lpad, nlaunches=5, sleep_time=0)")
                else:
                    f.write("    # rapidfire(lpad, nlaunches=5, sleep_time=0)")

            except Exception:
                raise
            print("\nFile successfully created!\nFind your auto sled at "
                  "{}\n".format(filename))

    else:
        raise ValueError("Please specify a callable function or a properly"
                         "formatted string location of the function")
    return filename

def PyTask_as_string(funcpath):
    return "PyTask(func=" + "'{}'".format(funcpath) + \
           ", args=[x], outputs=['_y_opt'])"

def OptTask_as_string(**kwargs):
    otstr = "OptTask("
    strlist = (str, unicode) if sys.version_info[0] < 3 else str
    for k, v in kwargs.items():
        if isinstance(v, strlist):
            strv = "'{}'".format(str(v))
        else:
            strv = str(v)
        otstr += str(k) + "=" + strv + ", "
    otstr = otstr[:-2] + ")"
    return otstr

if __name__ == "__main__":
    auto_setup(OptTask_as_string, [1,2,3])