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
        wf_creator (str): Module path to a function that returns a workflow
            based on a unique vector, x.
        dimensions ([tuple]): each 2-tuple in the list defines one dimension in
            the search space in (low, high) format.
            For categorical or discontinuous dimensions, includes all possible
            categories or values as a list of any length or a tuple of length>2.
            Example: dimensions = dim = [(1,100), (9.293, 18.2838), ("red",
            "blue", "green")].
        launchpad:
        **kwargs: Keyword arguments for defining the optimization. A full list
            of possible kwargs is given below:

            Predictors:
            predictor (string): names a function which given a list of explored
                points and unexplored points, returns an optimized guess.
                Included sklearn predictors are:
                    'GaussianProcessRegressor',
                    'RandomForestRegressor',
                    'ExtraTreesRegressor',
                    'GradientBoostingRegressor',
                To use a random guess, use 'random'
                Defaults to 'GaussianProcess'
                Example builtin predictor: predictor = 'GaussianProcessRegressor'
                Example custom predictor: predictor = 'my_module.my_predictor'
            predictor_args (list): the positional args to be passed to the model
                along with a list of points to be searched. For sklearn-based
                predictors included in OptTask, these positional args are passed to
                the init method of the chosen model. For custom predictors, these
                are passed to the chosen predictor function alongside the searched
                guesses, the output from searched guesses, and an unsearched space
                to be used with optimization.
            predictor_kwargs (dict): the kwargs to be passed to the model. Similar
                to predictor_args.

            Predictor performance:
            n_searchpts (int): The number of points to be searched in the search
                space when choosing the next best point. Choosing more points to
                search may increase the effectiveness of the optimization but take
                longer to evaluate. The default is 1000 points.
            n_trainpts (int): The number of already explored points to be chosen
                for training. Default is None, meaning all available points will be
                used for training. Reduce the number of points to decrease training
                times.
            space_file (str): The fully specified path of a pickle file containing a
                list of all possible searchable vectors.
                For example '/Users/myuser/myfolder/myspace.p'. When loaded, this
                space_file should be a list of tuples.
            acq (str): The acquisition function to use. Can be 'ei' for expected
                improvement, 'pi' for probability of improvement, or 'lcb' for lower
                confidence bound. Defaults to None, which means no acquisition
                function is used, and the highest predicted point is picked
                (greedy algorithm). Only applies to builtin predictors.
            n_bootstraps (int): The number of times each optimization should, sample,
                train, and predict values when generating uncertainty estimates for
                prediction. Only used if acq specified. At least 10 data points must
                be present for bootstrapping.

            Features:
            get_z (string): the fully-qualified name of a function which, given a x
                vector, returns another vector z which provides extra information
                to the machine learner. The features defined in z are not used to
                run the workflow, but are used for learning. If z_features are
                enabled, ONLY z features will be used for learning (x vectors
                essentially become tags or identifiers only).
                Examples:
                    get_z = 'my_module.my_fun'
                    get_z = '/path/to/folder/containing/my_package/my_module.my_fun'
            get_z_args (list): the positional arguments to be passed to the get_z
                function alongside x
            get_z_kwargs (dict): the kwargs to be passed to the get_z function
                alongside x
            persistent_z (str): The filename (pickle file) which should be used to
                store persistent z calculations. Specify this argument if
                calculating z for many (n_search_points) is not trivial and will
                cost time in computing. With this argument specified, each z will
                only be calculated once. Defaults to None, meaning that all
                unexplored z are re-calculated each iteration.
                Example:
                    persistent_z = '/path/to/persistent_z_guesses.p'

            Miscellaneous:
            wf_creator_args (list): the positional args to be passed to the
                wf_creator function alongsize the new x vector
            wf_creator_kwargs (dict): details the kwargs to be passed to the
                wf_creator function alongside the new x vector
            encode_categorical (bool): If True, preprocesses categorical data
                (strings) to one-hot encoded binary arrays for use with custom
                predictor functions. Default False.
            duplicate_check (bool): If True, checks that custom optimizers are not
                making duplicate guesses; all built-in optimizers cannot duplicate
                guess. If the custom predictor suggests a duplicate, OptTask picks
                a random guess out of the remaining untried space. Default is no
                duplicate check, and an error is raised if a duplicate is suggested.
            tolerances (list): The tolerance of each feature when duplicate
                checking. For categorical features, put 'None'
                Example: Our dimensions are [(1, 100), ['red', 'blue'],
                (2.0, 20.0)]. We want our first parameter to be  a duplicate only
                if it is exact, and our third parameter to be a duplicate if it is
                within 1e-6. Then:
                    tolerances=[0, None, 1e-6]
            maximize (bool): If true, makes optimization tend toward maximum values
                instead of minimum ones.
            batch_size (int): The number of jobs to submit per batch for a batch
                optimization. For example, batch_size=5 will optimize every 5th job,
                then submit another 5 jobs based on the best 5 predictions.
            enforce_sequential (bool): WARNING: Experimental feature! If True,
                enforces that RS optimizations are run sequentially (default), which
                prevents duplicate guesses from ever being run. If False, allows
                OptTasks to run optimizations in parallel, which may cause duplicate
                guesses with high parallelism.
            timeout (int): The number of seconds to wait before resetting the lock
                on the db.

    Returns:
        None: If you want to run the OptTask workflow, you'll need to pass in
        the launchpad and opt_label arguments in your wf_creator.

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
    acq_funcs = [None, 'ei', 'pi', 'lcb', 'maximin']
    if config['acq'] not in acq_funcs:
        raise ValueError(
            "Invalid acquisition function. Use 'ei', 'pi', 'lcb', "
            "'maximin' (multiobjective), or None.")

    # Insert config document
    config["doctype"] = "config"
    c = getattr(launchpad.db,  config["opt_label"])
    if c.find_one({"doctype": "config"}):
        opt_label = config["opt_label"]
        raise ValueError("A config is already present in this Launchpad for "
                         "opt_label=={}. Please use the reset function to reset"
                         " the database config.".format(opt_label))
    else:
        c.insert_one(config)
        logger = get_fw_logger("rocketsled", )
        logger.info("Rocketsled configuration succeeded.")


def reset(launchpad, opt_label="opt_default", delete=False):
    """
    Reset the optimization collection.

    Args:
        launchpad (LaunchPad): Fireworks LaunchPad object.
        opt_label (str): The label defining the optimization collection.
        delete (bool): If True, delete the collection. Otherwise, throw warning.

    Returns:
        None
    """
    c = getattr(launchpad.db, opt_label)
    if delete:
        c.delete_many({})
        logger = get_fw_logger("rocketsled")
        logger.info("Optimization collection reset.")
    else:
        warnings.warn("Set delete=True to reset the optimization collection.")


if __name__ == "__main__":
    from fireworks import LaunchPad
    # setup_config('somemod.something', [(1, 2), ["red", "green"]], LaunchPad(name="rsled"))
    reset(LaunchPad(name="rsled"), delete=True)