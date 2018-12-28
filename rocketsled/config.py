"""
Setup for the manager document.
"""
import warnings

from fireworks.utilities.fw_utilities import get_fw_logger

from rocketsled.task import OptTask
from rocketsled.utils import get_default_opttask_kwargs, check_dims, \
    is_discrete, deserialize

IMPORT_WARNING = "could not be imported! try putting it in a python package " \
                 "registered with PYTHONPATH or using the alternative " \
                 "syntax: /path/to/my/module.my_wfcreator"


class RailsConfig:
    """
    A class for configuring a rocketsled run.

    Args:
        wf_creator (str): Module path to a function that returns a workflow
            based on a unique vector, x.
        dimensions ([tuple]): each 2-tuple in the list defines one dimension in
            the search space in (low, high) format.
            For categorical or discontinuous dimensions, includes all possible
            categories or values as a list of any length or a tuple of length>2.
            Example: dimensions = dim = [(1,100), (9.293, 18.2838), ("red",
            "blue", "green")].
        launchpad (LaunchPad): The launchpad to use for storing optimization
            information.
        **kwargs: Keyword arguments for defining the optimization. A full list
            of possible kwargs is given below:

            Optimization data:
            opt_label (str): The label to use for this collection of
                optimization data.

            Workflow creator function:
            wf_creator_args (list): the positional args to be passed to the
                wf_creator function alongsize the new x vector
            wf_creator_kwargs (dict): details the kwargs to be passed to the
                wf_creator function alongside the new x vector

            Predictors (optimization):
            predictor (string): names a function which given a list of explored
                points and unexplored points, returns an optimized guess.
                Builtin sklearn-based predictors are:
                    'GaussianProcessRegressor',
                    'RandomForestRegressor',
                    'ExtraTreesRegressor',
                    'GradientBoostingRegressor',
                To use a random guess, use 'random'
                Defaults to 'GaussianProcess'
                Ex. builtin predictor: 'GaussianProcessRegressor'
                Ex. custom predictor: 'my_pkg.my_module.my_predictor'
            predictor_args (list): the positional args to be passed to the model
                along with a list of points to be searched. For sklearn-based
                predictors included in OptTask, these positional args are passed
                to the init method of the chosen model. For custom predictors,
                these are passed to the chosen predictor function alongside the
                searched guesses, the output from searched guesses, and an
                unsearched space to be used with optimization.
            predictor_kwargs (dict): the kwargs to be passed to the model.
                Similar to predictor_args.
            n_search_pts (int): The number of points to be searched in the
                search space when choosing the next best point. Choosing more
                points to search may increase the effectiveness of the
                optimization but take longer to evaluate. The default is 1000.
            n_train_pts (int): The number of already explored points to be
                chosen for training. Default is None, meaning all available
                points will be used for training. Reduce the number of points to
                decrease training times.
            n_bootstraps (int): The number of times each optimization should,
                sample, train, and predict values when generating uncertainty
                estimates for prediction. Only used if acq specified. At least
                10 data points must be present for bootstrapping.
            acq (str): The acquisition function to use. Can be 'ei' for expected
                improvement, 'pi' for probability of improvement, or 'lcb' for
                lower confidence bound. Defaults to None, which means no
                acquisition function is used, and the highest predicted point is
                picked (greedy algorithm). Only applies to builtin predictors.
            space_file (str): The fully specified path of a pickle file
                containing a list of all possible searchable vectors.
                For example '/Users/myuser/myfolder/myspace.p'. When loaded,
                this space_file should be a list of tuples.
            encode_categorical (bool): If True, preprocesses categorical data
                (strings) to one-hot encoded binary arrays for use with custom
                predictor functions. Default False.
            duplicate_check (bool): If True, checks that custom optimizers are
                not making duplicate guesses; all built-in optimizers cannot
                duplicate guess. If the custom predictor suggests a duplicate,
                OptTask picks a random guess out of the remaining untried space.
                Default is no duplicate check, and an error is raised if a
                duplicate is suggested.
            tolerances (list): The tolerance of each feature when duplicate
                checking. For categorical features, put 'None'
                Example: Our dimensions are [(1, 100), ['red', 'blue'],
                (2.0, 20.0)]. We want our first parameter to be  a duplicate
                only if it is exact, and our third parameter to be a duplicate
                if it is within 1e-6. Then:
                    tolerances=[0, None, 1e-6]

            z-vector features:
            get_z (string): the fully-qualified name of a function which, given
                an x vector, returns another vector z which provides extra
                information to the machine learner. The features defined in z
                are not used to run the workflow, but are used for learning. If
                z_features are enabled, ONLY z features will be used for
                learning (x vectors essentially become tags or identifiers only)
                Examples:
                    get_z = 'my_pkg.my_module.my_fun'
                    get_z = '/path/to/folder/containing/my_dir/my_module.my_fun'
            get_z_args (list): the positional arguments to be passed to the
                get_z function alongside x
            get_z_kwargs (dict): the kwargs to be passed to the get_z function
                alongside x
            z_file (str): The filename (pickle file) where OptTask should save
                /cache z calculations. Specify this argument if calculating z
                for many (n_search_pts) is not trivial and will cost time in
                computing. With this argument specified, each z  will only be
                calculated once. Defaults to None, meaning that all unexplored z
                are re-calculated each iteration.
                Example:
                    z_file = '/path/to/z_guesses.p'

            Parallelism:
            enforce_sequential (bool): WARNING: Experimental feature! If True,
                enforces that RS optimizations are run sequentially (default),
                which prevents duplicate guesses from ever being run. If False,
                allows OptTasks to run optimizations in parallel, which may
                cause duplicate guesses with high parallelism.
            batch_size (int): The number of jobs to submit per batch for a batch
                optimization. For example, batch_size=5 will optimize every 5th
                job, then submit another 5 jobs based on the best 5 predictions
                (recomputing the acquisition function after each prediction).
            timeout (int): The number of seconds to wait before resetting the
                lock on the db.

        Returns:
            None: If you want to run the OptTask workflow, you'll need to pass
            in the launchpad and opt_label arguments in your wf_creator.
        """
    def __init__(self, wf_creator, dimensions, launchpad, **kwargs):
        config = get_default_opttask_kwargs()
        for kw in kwargs.keys():
            if kw not in config:
                raise KeyError(
                    "{} not a valid argument for setup_config. Choose "
                    "from: {}".format(kw, list(config.keys())))
            else:
                config[kw] = kwargs[kw]
        config["wf_creator"] = wf_creator
        config["dimensions"] = dimensions
        config["launchpad"] = launchpad.to_db_dict()
        self.launchpad = launchpad
        self.opt_label = config["opt_label"]
        self.c = getattr(self.launchpad.db, config["opt_label"])

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
        config["doctype"] = "config"
        self.config = config
        self.logger = get_fw_logger("rocketsled")

    def configure(self):
        """
        Set up the optimization config. Required before using OptTask.
        """
        if self.c.find_one({"doctype": "config"}):
            opt_label = self.config["opt_label"]
            raise ValueError("A config is already present in this Launchpad "
                             "for opt_label=={}. Please use the reset function "
                             "to reset the database config.".format(opt_label))
        else:
            self.c.insert_one(self.config)
            self.logger.info("Rocketsled configuration succeeded.")

    def reset(self, delete=False):
        """
        Reset (delete) this optimization collection.

        Args:
            delete (bool): If True, delete the collection. Otherwise, throw
            warning.

        Returns:
            None
        """
        if delete:
            self.c.delete_many({})
            logger = get_fw_logger("rocketsled")
            logger.info("Optimization collection reset.")
        else:
            warnings.warn("Set delete=True to reset the optimization "
                          "collection.")

    def get_task(self):
        """
        Return a preconfigured OptTask which can be inserted into a workflow.

        Returns:
            OptTask: An OptTask object.
        """
        return OptTask(launchpad=self.launchpad, opt_label=self.opt_label)