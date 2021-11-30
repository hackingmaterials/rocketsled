"""
A class to configure, manage, and analyze optimizations. Similar to the
LaunchPad for FireWorks.
"""
import datetime
import math
import time
import warnings

import numpy as np
from fireworks.utilities.fw_utilities import get_fw_logger
from matplotlib import pyplot as plt

from rocketsled.task import OptTask
from rocketsled.utils import (
    NotConfiguredError,
    check_dims,
    deserialize,
    dtypes,
    get_default_opttask_kwargs,
    get_len,
    is_discrete,
    latex_float,
    pareto,
    serialize,
)

IMPORT_WARNING = (
    "could not be imported! try putting it in a python package "
    "registered with PYTHONPATH or using the alternative "
    "syntax: /path/to/my/module.my_wfcreator"
)


class MissionControl:
    """
    A class for configuring and controlling rocketsled optimization.

    Args:
        launchpad (LaunchPad): The launchpad to use for storing optimization
            information.
        opt_label (str): The name of the collection where Rocketsled should
            keep optimization data (in the same db as the LaunchPad). Please use
            a new collection (ie no other documents are present in the
            collection).
    """

    def __init__(self, launchpad, opt_label):
        self.logger = get_fw_logger("rocketsled")
        self.launchpad = launchpad
        self.opt_label = opt_label
        self.c = getattr(self.launchpad.db, opt_label)

        # The optimization colleciton may already exist, so check for manager
        # documents in the case it has already been configured.
        docs = self.c.find({"doctype": "config"})
        docs = [doc for doc in docs]
        if len(docs) > 1:
            raise ValueError(
                "There is more than one manager doc in the collection! Please"
                "use MissionControl.reset to reset the database, or manually"
                "remove the unneeded manager document!"
            )
        elif len(docs) == 1:
            self.config = dict(docs[0])
            self.is_configured = True
        else:
            self.config = None
            self.is_configured = False

    @property
    def task(self):
        """
        Return a preconfigured OptTask which can be inserted into a workflow.
        Make sure to run .configure before using this task, otherwise your
        workflow optimization might not work!

        Returns:
            OptTask: An OptTask object.
        """
        if self.is_configured:
            return OptTask(launchpad=self.launchpad, opt_label=self.opt_label)
        else:
            self.logger.warn(
                "OptTask created before configuration. Did you"
                "run MissionControl.configure(...)?"
            )

    def configure(self, wf_creator, dimensions, **kwargs):
        """
        Set up the optimization config. Required before using OptTask, but only
        needs to be done once. To reconfigure, use MissionControl.reset and then
        use configure again.

        Defaults can be found in defaults.yaml.

        Args:

        wf_creator (function or str): The function object that creates the
            workflow based on a unique vector, x. Alternatively, the full string
            module path to that function, e.g. "mypkg.mymodule.my_wf_creator",
            which must importable and found in PYTHONPATH.
        dimensions ([tuple]): each 2-tuple in the list defines one dimension in
            the search space in (low, high) format.
            For categorical or discontinuous dimensions, includes all possible
            categories or values as a list of any length or a tuple of length>2.
            Example: dimensions = dim = [(1,100), (9.293, 18.2838), ("red",
            "blue", "green")].
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
            predictor (function or str): a function which given a list of
                searched points and unsearched points, returns an optimized
                guess.

                To use a builtin predictor, pass in one of:
                    'GaussianProcessRegressor',
                    'RandomForestRegressor',
                    'ExtraTreesRegressor',
                    'GradientBoostingRegressor',
                    'random' (random guess)
                The default is 'GaussianProcessRegressor'

                To use a custom predictor, pass in the function object.
                Alternatively, the full string module path to that function,
                e.g. "mypkg.mymodule.my_predictor", which must importable and
                found in PYTHONPATH.
                Example builtin predictor: 'GaussianProcessRegressor'
                Example custom predictor: my_predictor
                Example custom predictor 2: 'my_pkg.my_module.my_predictor'
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
                estimates for prediction. At least 10 data points must be
                present for bootstrapping. Not used if: acq not specified,
                custom predictor used, or GaussianProcessRegressor used.
            acq (str): The acquisition function to use. Can be 'ei' for expected
                improvement, 'pi' for probability of improvement, or 'lcb' for
                lower confidence bound, or None for greedy selection. Only works
                with builtin predictors.
            space_file (str): The fully specified path of a pickle file
                containing a list of all possible searchable vectors.
                For example '/Users/myuser/myfolder/myspace.p'. When loaded,
                this space_file should be a list of tuples.
            onehot_categorical (bool): If True, preprocesses categorical data
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
            maximize (bool): If True, maximizes the objective function instead
                of minimizing. Defaults to False, meaninng minimze.

            z-vector features:
            get_z (function or str): the fully-qualified name of a function
                (or function object itself) which, given an x vector, returns
                another vector z which provides extra information to the machine
                learner. The features defined in z are not used to run the
                workflow, but are used for learning. If z_features are enabled,
                ONLY z features will be used for learning (x vectors essentially
                become tags or identifiers only)
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
        config = get_default_opttask_kwargs()
        config["launchpad"] = self.launchpad.to_db_dict()
        config["opt_label"] = self.opt_label
        for kw in kwargs.keys():
            if kw not in config:
                raise KeyError(
                    "{} not a valid argument for setup_config. Choose "
                    "from: {}".format(kw, list(config.keys()))
                )
            elif kw in ["get_z", "predictor"]:
                if hasattr(kwargs[kw], "__call__"):
                    config[kw] = serialize(kwargs[kw])
                else:
                    config[kw] = kwargs[kw]
            else:
                config[kw] = kwargs[kw]
        if hasattr(wf_creator, "__call__"):
            wf_creator = serialize(wf_creator)
        config["wf_creator"] = wf_creator
        config["dimensions"] = dimensions

        # Determine data types of dimensions
        config["dim_types"] = check_dims(dimensions)
        config["is_discrete_any"] = is_discrete(dimensions, criteria="any")
        config["is_discrete_all"] = is_discrete(dimensions, criteria="all")

        # Ensure importable functions are importable
        try:
            deserialize(wf_creator)
        except ImportError as IE:
            self.logger.warn("wf_creator " + IMPORT_WARNING + "\n" + str(IE))
        try:
            pre = config["predictor"]
            if pre:
                if "." in pre:
                    deserialize(pre)
        except ImportError as IE:
            self.logger.warn("predictor " + IMPORT_WARNING + "\n" + str(IE))
        try:
            getz = config["get_z"]
            if getz:
                if "." in getz:
                    deserialize(getz)
        except ImportError as IE:
            self.logger.warn("get_z " + IMPORT_WARNING + "\n" + str(IE))

        # Ensure acquisition function is valid (for builtin predictors)
        acq_funcs = [None, "ei", "pi", "lcb", "maximin"]
        if config["acq"] not in acq_funcs:
            raise ValueError(
                "Invalid acquisition function. Use 'ei', 'pi', 'lcb', "
                "'maximin' (multiobjective), or None."
            )
        config["doctype"] = "config"
        self.config = config
        if self.c.find_one({"doctype": "config"}):
            raise ValueError(
                "A config is already present in this Launchpad "
                "for opt_label=={}. Please use the MissionControl"
                " reset method to reset the database config."
                "".format(self.opt_label)
            )
        else:
            self.c.insert_one(self.config)
            self.logger.info("Rocketsled configuration succeeded.")
        self.is_configured = True

    def reset(self, hard=False):
        """
        Reset (delete) this optimization configuration and/or collection.

        Soft reset (hard=False): Delete the configuration, but keep the
            optimization data. This is useful if you are changing optimizers
            and want to keep the previous data (recommended)
        Hard reset (hard=True): Delete all data from the collection, including
            optimizatiomn data. WARNING - THIS OPTION IS NOT REVERSIBLE!

        Args:
            hard (bool): Whether to do a hard or soft reset. If False, deletes
                only the configuration, leaving the previously stored
                optimization data. If True, deletes everything from the
                optimization collection.

        Returns:
            None
        """
        if hard:
            self.c.delete_many({})
        else:
            self.c.delete_many({"doctype": "config"})
            self.c.delete_many({"doctype": "manager"})
        resetstr = "hard" if hard else "soft"
        self.logger.info(
            "Optimization collection {} {} reset."
            "".format(self.opt_label, resetstr)
        )
        self.is_configured = False

    def plot(
        self,
        show_best=True,
        show_mean=True,
        latexify=False,
        font_family="serif",
        scale="linear",
        summarize=True,
        print_pareto=False,
    ):
        """
        Visualize the progress of an optimization.

        Args:
            show_best (bool): Point out the best point on legend and on plot. If
                more than one best point (i.e., multiple equal maxima), show
                them all. If multiobjective, shows best for each objective, and
                prints the best value and x for each objective.
            show_mean (bool): Show the mean and standard deviation for the
                guesses as the computations are carried out.
            latexify (bool): Use LaTeX for formatting.
            font_family (str): The font family to use for rendering. Choose from
                'serif', 'sans-serif', 'fantasy', 'monospace', or 'cursive'.
            scale (str): Whether to scale the plot's y axis according to log
                ('log') or 'linear' scale.
            summarize (bool): If True, stdouts summary from .summarize.
            print_pareto (bool): If True, display all Pareto-optimal objective
                values.

        Returns:
            A matplotlib plot object handle
        """
        if not self.is_configured:
            raise NotConfiguredError(
                "Use MissionControl.configure to configure"
                "your optimization collection before "
                "plotting!"
            )
        maximize = self.config["maximize"]
        fxstr = "$f(x)$" if latexify else "f(x)"
        opt = max if maximize else min
        objs = self.c.find_one({"index": {"$exists": 1}})["y"]
        n_objs = len(objs) if isinstance(objs, (list, tuple)) else 1
        dt = datetime.datetime.now()
        dtdata = [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second]
        timestr = "{}-{}-{} {}:{}.{}".format(*dtdata)
        t0 = time.time()
        if latexify:
            plt.rc("text", usetex=True)
        else:
            plt.rc("text", usetex=False)
        plt.rc("font", family=font_family, size=9)
        n_cols = 3
        if n_objs < n_cols:
            _, ax_arr = plt.subplots(n_objs, squeeze=False)
        else:
            _, ax_arr = plt.subplots(
                n_cols, int(math.ceil(n_objs / n_cols)), squeeze=False
            )
        docset = self.c.find({"index": {"$exists": 1}})
        docs = [None] * docset.count()
        for i, doc in enumerate(docset):
            docs[i] = {"y": doc["y"], "index": doc["index"], "x": doc["x"]}
        if n_objs > 1:
            all_y = np.asarray([doc["y"] for doc in docs])
            pareto_set = all_y[pareto(all_y, maximize=maximize)].tolist()
            pareto_graph = [
                (i + 1, doc["y"])
                for i, doc in enumerate(docs)
                if doc["y"] in pareto_set
            ]
            pareto_i = [i[0] for i in pareto_graph]

        print("Optimization Analysis:")
        print("Number of objectives: {}".format(n_objs))

        for obj in range(n_objs):
            ax = ax_arr[obj % n_cols, int(math.floor(obj / n_cols))]

            i = []
            fx = []
            best = []
            mean = []
            std = []
            n = self.c.find().count() - 2

            for doc in docs:
                fx.append(doc["y"] if n_objs == 1 else doc["y"][obj])
                i.append(doc["index"])
                best.append(opt(fx))
                mean.append(np.mean(fx))
                std.append(np.std(fx))

            if time.time() - t0 > 60:
                self.logger.warn(
                    "Gathering data from the db is taking a while. Ensure"
                    "the latency to your db is low and the bandwidth"
                    "is as high as possible!"
                )

            mean = np.asarray(mean)
            std = np.asarray(std)

            ax.scatter(i, fx, color="blue", label=fxstr, s=10)
            ax.plot(
                i,
                best,
                color="orange",
                label="best {} value found so far" "".format(fxstr),
            )
            if show_mean:
                ax.plot(
                    i,
                    mean,
                    color="grey",
                    label="mean {} value (with std " "dev.)".format(fxstr),
                )
                ax.fill_between(i, mean + std, mean - std, color="grey", alpha=0.3)

            ax.set_xlabel("{} evaluation".format(fxstr))
            ax.set_ylabel("{} value".format(fxstr))
            best_val = opt(best)

            if show_best:

                if latexify:
                    best_label = "Best value: $f(x) = {}$" "".format(
                        latex_float(best_val)
                    )
                else:
                    best_label = "Best value: f(x) = {:.2E}".format(best_val)
                best = self.c.find({"y": best_val})

                if n_objs == 1:
                    print("\tNumber of optima: {}".format(best.count()))
                else:
                    print(
                        "\tNumber of optima for objective {}: {}"
                        "".format(obj + 1, best.count())
                    )

                for b in best:
                    bl = None if n_objs > 1 else best_label
                    ax.scatter(
                        [b["index"]],
                        [best_val],
                        color="darkgreen",
                        s=50,
                        linewidth=3,
                        label=bl,
                        facecolors="none",
                        edgecolors="darkgreen",
                    )

                    artext = "$x = $ [" if latexify else "x = ["
                    for i, xi in enumerate(b["x"]):
                        if i > 0:
                            artext += ". \mbox{~~~~~}" if latexify else "     "
                        if type(xi) in dtypes.floats:
                            if latexify:
                                artext += "${}$,\n".format(latex_float(xi))
                            else:
                                artext += "{:.2E},\n".format(xi)
                        else:
                            artext += str(xi) + ",\n"

                    artext = artext[:-2] + "]"
                    objstr = "objective {}".format(obj + 1) if n_objs > 1 else ""
                    if maximize:
                        print(
                            "\t\tmax(f(x)) {} is {} at x = {}"
                            "".format(objstr, best_val, b["x"])
                        )
                    else:
                        print(
                            "\t\tmin(f(x)) {} is {} at x = {}"
                            "".format(objstr, best_val, b["x"])
                        )
                    ax.annotate(
                        artext,
                        xy=(b["index"] + 0.5, best_val),
                        xytext=(b["index"] + float(n) / 12.0, best_val),
                        arrowprops=dict(color="green"),
                        color="darkgreen",
                        bbox=dict(facecolor="white", alpha=1.0),
                    )
            else:
                best_label = ""
            if n_objs > 1:
                pareto_fx = [i[1][obj] for i in pareto_graph]
                ax.scatter(
                    pareto_i, pareto_fx, color="red", label="Pareto optimal", s=20
                )
            if n_objs > 1:
                ax.set_title("Objective {}: {}".format(obj + 1, best_label))
            ax.set_yscale(scale)
        plt.gcf().set_size_inches(10, 10)
        if summarize:
            print(self.summarize())

        if print_pareto and n_objs > 1:
            print(
                "Pareto Frontier: {} points, ranked by hypervolume".format(
                    len(pareto_set)
                )
            )
            pareto_y = [doc["y"] for doc in docs if doc["y"] in pareto_set]
            pareto_x = [doc["x"] for doc in docs if doc["y"] in pareto_set]

            # Order y by hypervolume
            hypervolumes = [np.prod(y) for y in pareto_y]
            pareto_y_ordered = [
                y for _, y in sorted(zip(hypervolumes, pareto_y), reverse=True)
            ]
            pareto_x_ordered = [
                x for _, x in sorted(zip(hypervolumes, pareto_x), reverse=True)
            ]
            hypervolumes_ordered = sorted(hypervolumes, reverse=True)

            for i, _ in enumerate(pareto_set):
                print(
                    "f(x) = {} @ x = {} with hypervolume {}".format(
                        pareto_y_ordered[i],
                        pareto_x_ordered[i],
                        hypervolumes_ordered[i],
                    )
                )
        if n_objs % n_cols != 0 and n_objs > n_cols:
            for i in range(n_objs % n_cols, n_cols):
                plt.delaxes(ax_arr[i, -1])
        plt.legend()
        # plt.tight_layout(pad=0.01, w_pad=0.01, h_pad=0.01)
        plt.subplots_adjust(wspace=0.3, hspace=0.5)
        plt.suptitle(
            "Rocketsled optimization results for {} - "
            "{}".format(self.c.name, timestr),
            y=0.99,
        )
        return plt

    def summarize(self):
        """
        Returns stats about the optimization collection and checks consistency
        of the collection.

        Returns:
            fmtstr (str): The formatted information from the analysis, to print.
        """

        manager = self.c.find_one({"lock": {"$exists": 1}})
        qlen = len(manager["queue"])
        lock = manager["lock"]
        predictors = {}
        for doc in self.c.find(
            {"index": {"$exists": 1}, "y": {"$exists": 1, "$ne": "reserved"}}
        ):
            p = doc["predictor"]
            if p in predictors:
                predictors[p] += 1
            else:
                predictors[p] = 1
        dimdoc = self.c.find_one(
            {"index": {"$exists": 1}, "y": {"$exists": 1, "$ne": "reserved"}}
        )
        xdim = [type(d) for d in dimdoc["x"]]
        zdim = [type(d) for d in dimdoc["z"]]
        n_opts = sum(predictors.values())
        n_reserved = self.c.find({"y": "reserved"}).count()
        breakdown = ""
        for p, v in predictors.items():
            p = "No predictor" if not p else p
            predfrac = float(v) / float(n_opts)
            breakdown += "    * {0:.2f}%: ".format(predfrac * 100.0) + p + "\n"

        if not lock:
            lockstr = "DB not locked by any process (no current optimization)."
        else:
            lockstr = "DB locked by PID {}".format(lock)
        zlearn = "" if not zdim else "Only Z data is being used for learning."
        fmtstr = (
            "\nProblem dimension: \n    * X dimensions ({}): {}\n"
            "    * Z dimensions ({}): {}\n"
            "{}\n"
            "Number of Optimizations: {}\n"
            "Optimizers used (by percentage of optimizations): \n{}"
            "Number of reserved guesses: {}\n"
            "Number of waiting optimizations: {}\n"
            "{}\n".format(
                len(xdim),
                xdim,
                len(zdim),
                zdim,
                zlearn,
                n_opts,
                breakdown,
                n_reserved,
                qlen,
                lockstr,
            )
        )
        return fmtstr

    def fetch_matrices(self, include_reserved=False):
        """
        Return the X and Y matrices for this optimization.

        Args:
            include_reserved (bool): If True, returns "reserved" guesses (those
                which have been submitted to the launchpad but have not been
                successfully run). y values for these guesses are "reserved".


        Returns:
            all_x, all_y ([list], [list]): The X (input) matrix has dimensions
                n_samples, n_dimensions. The Y (output) matrix has dimensions
                n_samples, n_objectives. Only completed entries are retrieved.

        """
        if include_reserved:
            completed_query = {"y": {"$exists": 1}, "x": {"$exists": 1}}
        else:
            completed_query = {
                "y": {"$exists": 1, "$ne": "reserved"},
                "x": {"$exists": 1},
            }
        n_samples = self.c.count_documents(completed_query)
        all_x = [None] * n_samples
        all_y = [None] * n_samples

        n_objectives = get_len(self.c.find_one(completed_query)["y"])
        n_dimensions = len(self.c.find_one({"doctype": "config"})["dimensions"])
        dimension_mismatch = False
        objective_mismatch = False
        for i, doc in enumerate(self.c.find(completed_query)):
            all_x[i] = doc["x"]
            all_y[i] = doc["y"]
            if get_len(doc["x"]) != n_dimensions and not dimension_mismatch:
                dimension_mismatch = True
            if (
                get_len(doc["y"]) != n_objectives
                and not objective_mismatch
                and doc["y"] != "reserved"
            ):
                objective_mismatch = True
        if dimension_mismatch:
            warnings.warn(
                "Some entries have different dimensions from configuration "
                "({}). This optimization collection may be broken!"
                "".format(n_dimensions)
            )
        if objective_mismatch:
            warnings.warn(
                "Different numbers of objectives found. This "
                "optimization collectiomn may be broken!"
            )
        return all_x, all_y
