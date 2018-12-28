"""
Setup for the manager document.
"""
import warnings

from fireworks.utilities.fw_utilities import get_fw_logger

from rocketsled.utils import get_default_opttask_kwargs, check_dims, \
    is_discrete, deserialize

IMPORT_WARNING = "could not be imported! try putting it in a python package " \
                 "registered with PYTHONPATH or using the alternative " \
                 "syntax: /path/to/my/module.my_wfcreator"


def setup_config(wf_creator, dimensions, launchpad, **kwargs):
    """
    Set up the optimization config. Required before using OptTask.

    Args:
        wf_creator:
        dimensions:
        launchpad:
        **kwargs:

    Returns:
        None: If you want to run the OptTask workflow, you'll
            need to pass in the lpad and opt_label arguments in your wf_creator.

    """
    config = get_default_opttask_kwargs()
    for kw in kwargs.keys():
        if kw not in config:
            raise KeyError("{} not a valid argument for setup_config. Choose "
                           "from: {}".format(kw, list(config.keys())))
        else:
            config[kw] = kwargs[kw]
    config["wf_creator"] = wf_creator
    config["dimensions"] = dimensions
    config["launchpad"] = launchpad.to_db_dict()

    # Determine data types of dimensions
    config["dim_types"] = check_dims(dimensions)
    config["is_discrete_any"] = is_discrete(dimensions, criteria="any")
    config["is_discrete_all"] = is_discrete(dimensions, criteria="all")

    # Ensure importable functions are importable
    try:
        deserialize(wf_creator)
    except ImportError as IE:
        warnings.warn("wf_creator " + IMPORT_WARNING + "\n" + str(IE))
    try:
        pre = config["predictor"]
        if pre:
            if "." in pre:
                deserialize(pre)
    except ImportError as IE:
        warnings.warn("predictor " + IMPORT_WARNING + "\n" + str(IE))
    try:
        getz = config["get_z"]
        if getz:
            if "." in getz:
                deserialize(getz)
    except ImportError as IE:
        warnings.warn("get_z " + IMPORT_WARNING + "\n" + str(IE))

    # Ensure acquisition function is valid (for builtin predictors)
    if 'acq' in kwargs.keys():
        acq_funcs = [None, 'ei', 'pi', 'lcb', 'maximin']
        if kwargs['acq'] not in acq_funcs:
            raise ValueError(
                "Invalid acquisition function. Use 'ei', 'pi', 'lcb', "
                "'maximin' (multiobjective), or None.")

    for kwname, kwdict in \
            {'get_z_kwargs': get_z_kwargs,
             'predictor_kwargs': predkwargs}.items():
        if not isinstance(kwdict, dict):
            raise TypeError("{} should be a dictonary of keyword arguments."
                            "".format(kwname))

    for argname, arglist in \
            {'get_z_args': get_z_args,
             'predictor_args': predargs}.items():
        if not isinstance(arglist, (list, tuple)):
            raise TypeError("{} should be a list/tuple of positional "
                            "arguments".format(argname))

    # Insert config document
    config["doctype"] = "config"
    c = getattr(launchpad.db,  config["opt_label"])
    c.insert_one(config)
    logger = get_fw_logger("rocketsled")
    logger.info("Rocketsled configuration succeeded.")


if __name__ == "__main__":
    from fireworks import LaunchPad
    setup_config('somemod.something', [(1, 2), ["red", "green"]], LaunchPad(name="rsled"))
