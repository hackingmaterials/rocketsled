"""
The FireTask for running automatic optimization loops.

Please see the documentation for a comprehensive guide on usage.
"""
import pickle
import random
import warnings
from itertools import product
from os import getpid, path
from socket import gethostname
from time import sleep

import numpy as np
import tqdm
from fireworks import FWAction, LaunchPad
from fireworks.core.firework import FireTaskBase
from fireworks.utilities.fw_utilities import explicit_serialize
from sklearn.preprocessing import LabelBinarizer, StandardScaler

from rocketsled.acq import acquire, predict
from rocketsled.utils import (
    BUILTIN_PREDICTORS,
    BatchNotReadyError,
    ExhaustedSpaceError,
    NotConfiguredError,
    ObjectiveError,
    convert_native,
    deserialize,
    dtypes,
    is_duplicate_by_tolerance,
    pareto,
    split_xz,
)

__author__ = "Alexander Dunn"
__email__ = "ardunn@lbl.gov"


@explicit_serialize
class OptTask(FireTaskBase):
    """
    A FireTask for automatically running optimization loops and storing
    optimization data for complex workflows.

    OptTask takes in _x and _y from the fw_spec (input/output of
    current guess), gathers X (previous guesses input) and y (previous guesses
    output), and predicts the next best guess.

    Required args:
        launchpad (LaunchPad): A Fireworks LaunchPad object, which can be used
            to define the host/port/name of the db.
        opt_label (string): Names the collection of that the particular
            optimization's data will be stored in. Multiple collections
            correspond to multiple independent optimizations.
    """

    _fw_name = "OptTask"
    required_params = ["launchpad", "opt_label"]

    def __init__(self, *args, **kwargs):
        super(OptTask, self).__init__(*args, **kwargs)

        # Configuration attrs
        lp = self.get("launchpad", LaunchPad.auto_load())
        if isinstance(lp, LaunchPad):
            lp = lp.to_dict()
        self.lpad = LaunchPad.from_dict(lp)
        self.opt_label = self.get("opt_label", "opt_default")
        self.c = getattr(self.lpad.db, self.opt_label)
        self.config = self.c.find_one({"doctype": "config"})
        if self.config is None:
            raise NotConfiguredError(
                "Please use MissionControl().configure to "
                "configure the optimization database "
                "({} - {}) before running OptTask."
                "".format(self.lpad.db, self.opt_label)
            )
        self.wf_creator = deserialize(self.config["wf_creator"])
        self.x_dims = self.config["dimensions"]
        self._xdim_types = self.config["dim_types"]
        self.is_discrete_all = self.config["is_discrete_all"]
        self.is_discrete_any = self.config["is_discrete_any"]
        self.wf_creator_args = self.config["wf_creator_args"] or []
        self.wf_creator_kwargs = self.config["wf_creator_kwargs"] or {}
        self.predictor = self.config["predictor"]
        self.predictor_args = self.config["predictor_args"] or []
        self.predictor_kwargs = self.config["predictor_kwargs"] or {}
        self.maximize = self.config["maximize"]
        self.n_search_pts = self.config["n_search_pts"]
        self.n_train_pts = self.config["n_train_pts"]
        self.n_bootstraps = self.config["n_bootstraps"]
        self.acq = self.config["acq"]
        self.space_file = self.config["space_file"]
        self.onehot_categorical = self.config["onehot_categorical"]
        self.duplicate_check = self.config["duplicate_check"]
        self.get_z = self.config["get_z"]
        if self.get_z:
            self.get_z = deserialize(self.config["get_z"])
        else:
            self.get_z = lambda *ars, **kws: []
        self.get_z_args = self.config["get_z_args"] or []
        self.get_z_kwargs = self.config["get_z_kwargs"] or {}
        self.z_file = self.config["z_file"]
        self.enforce_sequential = self.config["enforce_sequential"]
        self.tolerances = self.config["tolerances"]
        self.batch_size = self.config["batch_size"]
        self.timeout = self.config["timeout"]

        # Declared attrs
        self.n_objs = None
        self.builtin_predictors = {p.__name__: p for p in BUILTIN_PREDICTORS}
        self._n_cats = 0
        self._encoding_info = []

        # Query formats
        self._completed = {
            "x": {"$exists": 1},
            "y": {"$exists": 1, "$ne": "reserved"},
            "z": {"$exists": 1},
        }
        self._manager = {"lock": {"$exists": 1}, "queue": {"$exists": 1}}

    def run_task(self, fw_spec):
        """
        FireTask for running an optimization loop.

        Args:
            fw_spec (dict): the firetask spec. Must contain a '_y' key with
            a float type field and must contain a '_x' key containing a
            vector uniquely defining the point in search space.

        Returns:
            (FWAction) A workflow based on the workflow creator and a new,
            optimized guess.
        """
        pid = f"{getpid()}@{gethostname()}"
        sleeptime = 0.01
        max_runs = int(self.timeout / sleeptime)
        max_resets = 3

        # Running stepwise optimization for concurrent processes requires a
        # manual 'lock' on the optimization database to prevent duplicate
        # guesses. The first process sets up a manager document which handles
        # locking and queueing processes by PID. The single, active process in
        # the lock is free to access optimization data; the queue of the manager
        # holds parallel process PIDs waiting to access the db. When the active
        # process finishes, it removes itself from the lock and moves the first
        # queue PID into the lock, allowing the next process to begin
        # optimization. Each process continually tries to either queue or place
        # itself into the lock if not active.

        for run in range(max_resets * max_runs):
            manager_count = self.c.count_documents(self._manager)
            if manager_count == 0:
                self.c.insert_one({"lock": pid, "queue": [], "doctype": "manager"})
            elif manager_count == 1:
                # avoid bootup problems if manager lock is being deleted
                # concurrently with this check
                try:
                    manager = self.c.find_one(self._manager)
                    manager_id = manager["_id"]
                    lock = manager["lock"]
                except TypeError:
                    continue

                if lock is None:
                    self.c.find_one_and_update(
                        {"_id": manager_id}, {"$set": {"lock": pid}}
                    )

                elif self.enforce_sequential and lock != pid:
                    if pid not in manager["queue"]:

                        # avoid bootup problems if manager queue is being
                        # deleted concurrently with this check
                        try:
                            self.c.find_one_and_update(
                                {"_id": manager_id}, {"$push": {"queue": pid}}
                            )
                        except TypeError:
                            continue
                    else:
                        sleep(sleeptime)
                elif not self.enforce_sequential or (
                    self.enforce_sequential and lock == pid
                ):
                    try:
                        x, y, z, all_xz_new, n_completed = self.optimize(
                            fw_spec, manager_id
                        )
                    except BatchNotReadyError:
                        return None
                    except Exception:
                        self.pop_lock(manager_id)
                        raise

                    # make sure a process has not timed out and changed the lock
                    # pid while this process is computing the next guess
                    try:
                        if (
                            self.c.find_one(self._manager)["lock"] != pid
                            or self.c.count_documents(self._manager) == 0
                        ):
                            continue
                        else:
                            opt_id = self.stash(x, y, z, all_xz_new, n_completed)
                    except TypeError as E:
                        warnings.warn(
                            "Process {} probably timed out while "
                            "computing next guess, with exception {}."
                            " Try shortening the training time or "
                            "lengthening the timeout for OptTask!"
                            "".format(pid, E),
                            RuntimeWarning,
                        )
                        raise E
                        # continue
                    self.pop_lock(manager_id)
                    all_x_new = [
                        split_xz(xz_new, self.x_dims, x_only=True)
                        for xz_new in all_xz_new
                    ]
                    if not isinstance(self.wf_creator_args, (list, tuple)):
                        raise TypeError(
                            "wf_creator_args should be a list/tuple of "
                            "positional arguments."
                        )

                    if not isinstance(self.wf_creator_kwargs, dict):
                        raise TypeError(
                            "wf_creator_kwargs should be a dictionary of "
                            "keyword arguments."
                        )

                    new_wfs = [
                        self.wf_creator(
                            x_new, *self.wf_creator_args, **self.wf_creator_kwargs
                        )
                        for x_new in all_x_new
                    ]
                    self.lpad.bulk_add_wfs(new_wfs)
                    return FWAction(
                        update_spec={"_optimization_id": opt_id},
                        stored_data={"_optimization_id": opt_id},
                    )
            else:
                # Delete the manager that this has created
                self.c.delete_one({"lock": pid})

            if run in [max_runs * k for k in range(1, max_resets)]:
                self.c.find_one_and_update(
                    self._manager, {"$set": {"lock": None, "queue": []}}
                )

            elif run == max_runs * max_resets:
                raise Exception(
                    "The manager is still stuck after "
                    "resetting. Make sure no stalled processes "
                    "are in the queue."
                )

    def optimize(self, fw_spec, manager_id):
        """
        Run the optimization algorithm.

        Args:
            fw_spec (dict): The firework spec.
            manager_id (ObjectId): The MongoDB object id of the manager
                document.

        Returns:
            x (iterable): The current x guess.
            y: (iterable): The current y (objective function) value
            z (iterable): The z vector associated with x
            all_xz_new ([list] or [tuple]): The predicted next best guess(es),
                including their associated z vectors
            n_completed (int): The number of completed guesses/workflows
        """
        x = list(fw_spec["_x"])
        y = fw_spec["_y"]
        if isinstance(y, (list, tuple)):
            if len(y) == 1:
                y = y[0]
            self.n_objs = len(y)
            if self.acq not in ("maximin", None):
                raise ValueError(
                    "{} is not a valid acquisition function for multiobjective "
                    "optimization".format(self.acq)
                )
        else:
            if self.acq == "maximin":
                raise ValueError(
                    "Maximin is not a valid acquisition function for single "
                    "objective optimization."
                )
            self.n_objs = 1

        # If process A suggests a certain guess and runs it, process B may
        # suggest the same guess while process A is running its new workflow.
        # Therefore, process A must reserve the guess. Line below releases
        # reservation on this document in case of workflow failure or end of
        # workflow.
        self.c.delete_one({"x": x, "y": "reserved"})

        # fetch additional attributes for constructing ML model
        z = self.get_z(x, *self.get_z_args, **self.get_z_kwargs)

        # use all possible training points as default
        n_completed = self.c.count_documents(self._completed)
        if not self.n_train_pts or self.n_train_pts > n_completed:
            self.n_train_pts = n_completed

        # check if optimization should be done, if in batch mode
        batch_mode = False if self.batch_size == 1 else True
        batch_ready = (
            n_completed not in (0, 1) and (n_completed + 1) % self.batch_size == 0
        )

        x = convert_native(x)
        y = convert_native(y)
        z = convert_native(z)

        if batch_mode and not batch_ready:
            # 'None' predictor means this job was not used for
            # an optimization run.
            if self.c.find_one({"x": x}):
                if self.c.find_one({"x": x, "y": "reserved"}):
                    # For reserved guesses: update everything
                    self.c.find_one_and_update(
                        {"x": x, "y": "reserved"},
                        {
                            "$set": {
                                "y": y,
                                "z": z,
                                "z_new": [],
                                "x_new": [],
                                "predictor": None,
                                "index": n_completed + 1,
                            }
                        },
                    )
                else:
                    # For completed guesses (ie, this workflow
                    # is a forced duplicate), do not update
                    # index, but update everything else
                    self.c.find_one_and_update(
                        {"x": x},
                        {
                            "$set": {
                                "y": y,
                                "z": z,
                                "z_new": [],
                                "x_new": [],
                                "predictor": None,
                            }
                        },
                    )
            else:
                # For new guesses: insert x, y, z, index,
                # predictor, and dummy new guesses
                self.c.insert_one(
                    {
                        "x": x,
                        "y": y,
                        "z": z,
                        "x_new": [],
                        "z_new": [],
                        "predictor": None,
                        "index": n_completed + 1,
                    }
                )
            self.pop_lock(manager_id)
            raise BatchNotReadyError

        # Mongo aggregation framework may give duplicate documents, so we cannot
        # use $sample to randomize the training points used
        searched_indices = random.sample(range(1, n_completed + 1), self.n_train_pts)
        searched_docs = self.c.find(
            {"index": {"$in": searched_indices}}, batch_size=10000
        )
        reserved_docs = self.c.find({"y": "reserved"}, batch_size=10000)
        reserved = []
        for doc in reserved_docs:
            reserved.append(doc["x"])
        all_y = [None] * n_completed
        all_y.append(y)
        all_x_searched = [None] * n_completed
        all_x_searched.append(x)
        z = list(z)
        all_xz_searched = [None] * n_completed
        all_xz_searched.append(x + z)
        for i, doc in enumerate(searched_docs):
            all_x_searched[i] = doc["x"]
            all_xz_searched[i] = doc["x"] + doc["z"]
            all_y[i] = doc["y"]

        all_x_space = self._discretize_space(self.x_dims)
        all_x_space = list(all_x_space) if self.z_file else all_x_space
        all_x_unsearched = []
        for xi in all_x_space:
            xj = list(xi)
            if xj not in all_x_searched and xj not in reserved:
                all_x_unsearched.append(xj)
                if len(all_x_unsearched) == self.n_search_pts:
                    break

        if self.z_file:
            if path.exists(self.z_file):
                with open(self.z_file, "rb") as f:
                    xz_map = pickle.load(f)
            else:
                xz_map = {
                    tuple(xi): self.get_z(xi, *self.get_z_args, **self.get_z_kwargs)
                    for xi in all_x_space
                }
                with open(self.z_file, "wb") as f:
                    pickle.dump(xz_map, f)

            all_xz_unsearched = [xi + xz_map[tuple(xi)] for xi in all_x_unsearched]
        else:
            all_xz_unsearched = [
                xi + self.get_z(xi, *self.get_z_args, **self.get_z_kwargs)
                for xi in all_x_unsearched
            ]

        # if there are no more unsearched points in the entire
        # space, either they have been searched (ie have x, y,
        # and z) or have been reserved.
        if len(all_xz_unsearched) < 1:
            if self.is_discrete_all:
                raise ExhaustedSpaceError(
                    "The discrete space has been searched exhaustively."
                )
            else:
                raise TypeError(
                    "A comprehensive list of points was exhausted "
                    "but the dimensions are not discrete."
                )
        z_dims = self._z_dims(all_xz_unsearched, all_xz_searched)
        xz_dims = self.x_dims + z_dims

        # run machine learner on Z or X features
        if self.predictor in self.builtin_predictors:
            model = self.builtin_predictors[self.predictor]
            all_xz_searched = self._encode(all_xz_searched, xz_dims)
            all_xz_unsearched = self._encode(all_xz_unsearched, xz_dims)
            all_xz_new_onehot = []

            if self.batch_size > 1:
                iterator_obj = tqdm.tqdm(
                    range(self.batch_size), desc="Batch predictions"
                )
            else:
                iterator_obj = range(self.batch_size)
            for _ in iterator_obj:
                xz1h = self._predict(
                    all_xz_searched,
                    all_y,
                    all_xz_unsearched,
                    model(*self.predictor_args, **self.predictor_kwargs),
                    self.maximize,
                    scaling=True,
                )
                ix = all_xz_unsearched.index(xz1h)
                all_xz_unsearched.pop(ix)
                all_xz_new_onehot.append(xz1h)
            all_xz_new = [
                self._decode(xz_onehot, xz_dims) for xz_onehot in all_xz_new_onehot
            ]

        elif self.predictor == "random":
            all_xz_new = random.sample(all_xz_unsearched, self.batch_size)

        else:
            # If using a custom predictor, automatically convert
            # categorical info to one-hot encoded ints.
            # Used when a custom predictor cannot natively use
            # categorical info
            if self.onehot_categorical:
                all_xz_searched = self._encode(all_xz_searched, xz_dims)
                all_xz_unsearched = self._encode(all_xz_unsearched, xz_dims)

            try:
                predictor_fun = deserialize(self.predictor)
            except Exception as E:
                raise NameError(
                    "The custom predictor {} didnt import "
                    "correctly!\n{}".format(self.predictor, E)
                )

            all_xz_new = predictor_fun(
                all_xz_searched,
                all_y,
                self.x_dims,
                all_xz_unsearched,
                *self.predictor_args,
                **self.predictor_kwargs,
            )
            if self.onehot_categorical:
                all_xz_new = self._decode(all_xz_new, xz_dims)

            if not isinstance(all_xz_new[0], (list, tuple, np.ndarray)):
                all_xz_new = [all_xz_new]

        # duplicate checking for custom optimizer functions
        if self.duplicate_check:

            if not self.enforce_sequential:
                raise ValueError(
                    "Duplicate checking cannot work when "
                    "optimizations are not enforced sequentially."
                )
            if (
                self.predictor not in self.builtin_predictors
                and self.predictor != "random"
            ):
                all_x_new = [
                    split_xz(xz_new, self.x_dims, x_only=True)
                    for xz_new in all_xz_new
                ]
                all_x_searched = [
                    split_xz(xz, self.x_dims, x_only=True) for xz in all_xz_searched
                ]
                if self.tolerances:
                    for n, x_new in enumerate(all_x_new):
                        if is_duplicate_by_tolerance(
                            x_new, all_x_searched, tolerances=self.tolerances
                        ):
                            all_xz_new[n] = random.choice(all_xz_unsearched)
                else:
                    if self.is_discrete_all:
                        # test only for x, not xz because custom predicted z
                        # may not be accounted for
                        for n, x_new in enumerate(all_x_new):
                            if x_new in all_x_searched or x_new == x:
                                all_xz_new[n] = random.choice(all_xz_unsearched)
                    else:
                        raise ValueError(
                            "Define tolerances parameter to "
                            "duplicate check floats."
                        )
        return x, y, z, all_xz_new, n_completed

    def stash(self, x, y, z, all_xz_new, n_completed):
        """
        Write documents to database after optimization.

        Args:
            x (iterable): The current x guess.
            y: (iterable): The current y (objective function) value
            z (iterable): The z vector associated with x
            all_xz_new ([list] or [tuple]): The predicted next best guess(es),
                including their associated z vectors
            n_completed (int): The number of completed guesses/workflows

        Returns:
            opt_id (pymongo InsertedOneResult): The result of the insertion
                of the new optimization document in the database. If multiple
                opt_ids are valid (ie batch mode is enabled), the last opt_id
                is returned.
        """

        for xz_new in all_xz_new:
            # separate 'predicted' z features from the new x vector
            x_new, z_new = split_xz(xz_new, self.x_dims)
            x_new = convert_native(x_new)
            z_new = convert_native(z_new)

            # if it is a duplicate (such as a forced
            # identical first guess)
            forced_dupe = self.c.find_one({"x": x})

            acqmap = {
                "ei": "Expected Improvement",
                "pi": "Probability of Improvement",
                "lcb": "Lower Confidence Boundary",
                None: "Highest Value",
                "maximin": "Maximin Expected " "Improvement",
            }
            if self.predictor in self.builtin_predictors:
                predictorstr = (
                    self.predictor + " with acquisition: " + acqmap[self.acq]
                )
                if self.n_objs > 1:
                    predictorstr += " using {} objectives".format(self.n_objs)
            else:
                predictorstr = self.predictor
            if forced_dupe:
                # only update the fields which should be updated
                self.c.find_one_and_update(
                    {"x": x},
                    {
                        "$set": {
                            "y": y,
                            "z": z,
                            "z_new": z_new,
                            "x_new": x_new,
                            "predictor": predictorstr,
                        }
                    },
                )
            else:
                # update all the fields, as it is a new document
                self.c.insert_one(
                    {
                        "z": z,
                        "y": y,
                        "x": x,
                        "z_new": z_new,
                        "x_new": x_new,
                        "predictor": predictorstr,
                        "index": n_completed + 1,
                    }
                )
            # ensure previously fin. workflow results are not overwritten by
            # concurrent predictions
            if (
                self.c.count_documents(
                    {"x": x_new, "y": {"$exists": 1, "$ne": "reserved"}}
                )
                == 0
            ):
                # reserve the new x to prevent parallel processes from
                # registering it as unsearched, since the next iteration of this
                # process will be exploring it
                res = self.c.insert_one({"x": x_new, "y": "reserved"})
                opt_id = res.inserted_id
            else:
                raise ValueError(
                    "The predictor suggested a guess which has already been "
                    "tried: {}".format(x_new)
                )
        return opt_id

    def pop_lock(self, manager_id):
        """
        Releases the current process lock on the manager doc, and moves waiting
        processes from the queue to the lock.

        Args:
            manager_id: The MongoDB ObjectID object of the manager doc.

        Returns:
            None
        """
        queue = self.c.find_one({"_id": manager_id})["queue"]
        if not queue:
            self.c.find_one_and_update({"_id": manager_id}, {"$set": {"lock": None}})
        else:
            new_lock = queue.pop(0)
            self.c.find_one_and_update(
                {"_id": manager_id}, {"$set": {"lock": new_lock, "queue": queue}}
            )

    def _discretize_space(self, dims, n_floats=100):
        """
        Create a list of points for searching during optimization.

        Args:
            dims ([tuple]): dimensions of the search space.
            n_floats (int): Number of floating points to sample from each
                continuous dimension when discrete dimensions are present. If
                all dimensions are continuous, this argument is ignored and
                a space of n_searchpts is generated in a more efficient manner.

        Returns:
            ([list]) Points of the search space.
        """
        if "space_file" in self:
            if self["space_file"]:
                with open(self["space_file"], "rb") as f:
                    return pickle.load(f)

        # Ensure consistency of dimensions
        for dim in dims:
            if isinstance(dim, tuple) and len(dim) == 2:
                for dtype in ["ints", "floats"]:
                    if (
                        type(dim[0])
                        not in getattr(dtypes, dtype)
                        != type(dim[1])
                        not in getattr(dtypes, dtype)
                    ):
                        raise ValueError(
                            "Ranges of values for dimensions "
                            "must be the same general datatype,"
                            "not ({}, {}) for {}"
                            "".format(type(dim[0]), type(dim[1]), dim)
                        )

        dims_ranged = all([len(dim) == 2 for dim in dims])
        dims_float = all([type(dim[0]) in dtypes.floats for dim in dims])
        if dims_float and dims_ranged:
            # Save computation/memory if all ranges of floats
            nf = self.n_search_pts
            space = np.zeros((nf, len(dims)))
            for i, dim in enumerate(dims):
                low = dim[0]
                high = dim[1]
                space[:, i] = np.random.uniform(low=low, high=high, size=nf)
            return space.tolist()
        else:
            # todo: this could be faster
            total_dimspace = []
            for dim in dims:
                if isinstance(dim, (tuple, list)) and len(dim) == 2:
                    low = dim[0]
                    high = dim[1]
                    if type(low) in dtypes.ints:
                        # Then the dimension is of the form (low, high)
                        dimspace = list(range(low, high + 1))
                    elif type(low) in dtypes.floats:
                        dimspace = np.random.uniform(
                            low=low, high=high, size=n_floats
                        ).tolist()
                    else:  # The dimension is a 2-tuple of strings
                        dimspace = dim
                else:  # the dimension is a list of entries
                    dimspace = dim
                random.shuffle(dimspace)
                total_dimspace.append(dimspace)
            if len(dims) == 1:
                return [[xi] for xi in total_dimspace[0]]
            else:
                return product(*total_dimspace)

    def _predict(self, all_x, all_y, space, model, maximize, scaling):
        """
        Scikit-learn compatible model for stepwise optimization. It uses a
        regressive predictor evaluated on remaining points in a discrete space.

        Since sklearn modules cannot deal with categorical data, categorical
        data is preprocessed by _encode before being passed to _predict,
        and predicted x vectors are postprocessed by _decode to convert to
        the original categorical dimensions.

        Args:
            all_x ([list]): List of vectors containing input training data.
            all_y (list): List of scalars containing output training data. Can
                be a list of vectors if undergoing multiobjective optimization.
            space ([list]): List of vectors containing all unsearched inputs.
                Should be preprocessed.
            model (sklearn model): The regressor used for predicting the next
                best guess.
            maximize (bool): Makes predictor return the guess which maximizes
                the predicted objective function output. Else minmizes the
                predicted objective function output.
            scaling (bool): If True, scale data with StandardScaler (required
                for some optimizers, such as Gaussian processes).

        Returns:
            (list) A vector which is predicted to minimize (or maximize) the
                objective function.
        """

        # Scale data if all floats for dimensions in question
        if scaling:
            scaler = StandardScaler()
            train_set = np.vstack((all_x, space))
            scaler.fit(train_set)
            all_x_scaled = scaler.transform(all_x)
            space_scaled = scaler.transform(space)
        else:
            all_x_scaled = all_x
            space_scaled = space
        n_searched = len(all_x)
        n_unsearched = len(space)

        # If get_z defined, only use z features!
        if "get_z" in self:
            encoded_xlen = 0
            for t in self._xdim_types:
                if "int" in t or "float" in t:
                    encoded_xlen += 1
                else:
                    encoded_xlen += int(t[-1])
            all_x_scaled = np.asarray(all_x_scaled)[:, encoded_xlen:]
            space_scaled = np.asarray(space_scaled)[:, encoded_xlen:]
        all_y = np.asarray(all_y)
        if self.n_objs == 1:
            # Single objective
            if maximize:
                all_y = -1.0 * all_y
            if self.acq is None or n_searched < 10:
                model.fit(all_x_scaled, all_y)
                values = model.predict(space_scaled).tolist()
                evaluator = min
            else:
                # Use the acquistion function values

                predictions = predict(
                    all_x_scaled, all_y, space_scaled, model, self.n_bootstraps
                )
                mu, std = predictions
                values = acquire(self.acq, all_y, mu, std)
                evaluator = max
        else:
            evaluator = max
            # Multi-objective
            if self.acq is None or n_searched < 10:
                values = np.zeros((n_unsearched, self.n_objs))
                for i in range(self.n_objs):
                    yobj = [y[i] for y in all_y]
                    model.fit(all_x_scaled, yobj)
                    values[:, i] = model.predict(space_scaled)
                # In exploitative strategy, randomly weight pareto optimial
                # predictions!
                values = pareto(values, maximize=maximize) * np.random.uniform(
                    0, 1, n_unsearched
                )

            else:
                # Adapted from Multiobjective Optimization of Expensive Blackbox
                # Functions via Expected Maximin Improvement
                # by Joshua D. Svenson, Thomas J. Santner
                if maximize:
                    all_y = -1.0 * all_y

                if self.acq != "maximin":
                    raise ObjectiveError(
                        "Multiple objectives detected, but Maximin acquisition "
                        "function is not used. Please use a single objective "
                        "or change the acquisition function."
                    )
                mu = np.zeros((n_unsearched, self.n_objs))
                values = np.zeros((n_unsearched, self.n_objs))
                for i in range(self.n_objs):
                    yobj = [y[i] for y in all_y]
                    values[:, i], mu[:, i] = acquire(
                        "pi",
                        all_x_scaled,
                        yobj,
                        space_scaled,
                        model,
                        self.n_bootstraps,
                        return_means=True,
                    )
                pf = all_y[pareto(all_y, maximize=maximize)]
                dmaximin = np.zeros(n_unsearched)
                for i, mui in enumerate(mu):
                    mins = np.zeros(len(pf))
                    for j, pfj in enumerate(pf):
                        # select max distance to pareto point (improvements
                        # are negative) among objectives
                        mins[j] = min(mui - pfj)
                    # minimum among all pareto points of the maximum improvement
                    # among objectives. Min/max are reversed bc. minimization
                    dmaximin[i] = max(mins)

                if len(dmaximin[dmaximin < 0.0]) != 0:
                    # Predicted pareto-optimal solutions are negative so far
                    # If we are here, it means there are still predicted pareto
                    # optimal solutions. This procedure is as shown in original
                    # EI paper.
                    dmaximin = dmaximin * -1.0
                    dmaximin = dmaximin.clip(min=0)
                else:
                    # Addition if there are no predicted pareto solutions.
                    # Without this, all dmaximin values are zero if no predicted
                    # pareto solutions. With this, dmaximin values are inverted
                    # to find the 'least bad' non-pareto optimal value.
                    # Only using the 'if' block above will result in pure
                    # exploration (random) if no pareto-optimal solutions
                    # are predicted.
                    dmaximin = 1.0 / dmaximin

                pi_product = np.prod(values, axis=1)
                values = pi_product * dmaximin
            values = values.tolist()
        prediction = evaluator(values)
        index = values.index(prediction)
        return space[index]

    def _encode(self, all_x, dims):
        """
        Transforms data containing categorical information to "one-hot" encoded
        data, since sklearn cannot process categorical data on its own.

        Args:
            all_x ([list]): The search space, possibly containing categorical
                dimensions.
            dims: The dimensions of the search space. Used to define all
                possible choices for categorical dimensions so that categories
                are properly encoded.

        Returns:
            X ([list]): "One-hot" encoded forms of X data containing categorical
                dimensions. Search spaces which are  completely numerical are
                unchanged.
        """
        self._n_cats = 0
        self._encoding_info = []
        for i, dim in enumerate(dims):
            if type(dim[0]) in dtypes.others:
                cats = [0] * len(all_x)
                for j, x in enumerate(all_x):
                    cats[j] = x[i - self._n_cats]
                forward_map = {k: v for v, k in enumerate(dim)}
                inverse_map = {v: k for k, v in forward_map.items()}
                lb = LabelBinarizer()
                lb.fit([forward_map[v] for v in dim])
                binary = lb.transform([forward_map[v] for v in cats])
                for j, x in enumerate(all_x):
                    del x[i - self._n_cats]
                    x += list(binary[j])
                dim_info = {
                    "lb": lb,
                    "inverse_map": inverse_map,
                    "binary_len": len(binary[0]),
                }
                self._encoding_info.append(dim_info)
                self._n_cats += 1
        return all_x

    def _decode(self, new_x, dims):
        """
        Convert a "one-hot" encoded point (the predicted guess) back to the
        original categorical dimensions.

        Args:
            new_x (list): The "one-hot" encoded new x vector predicted by the
                predictor.
            dims ([list]): The dimensions of the search space.

        Returns:
            categorical_new_x (list): The new_x vector in categorical dimensions
        """

        original_len = len(dims)
        static_len = original_len - self._n_cats
        categorical_new_x = []
        cat_index = 0
        tot_bin_len = 0

        for i, dim in enumerate(dims):
            if type(dim[0]) in dtypes.others:
                dim_info = self._encoding_info[cat_index]
                binary_len = dim_info["binary_len"]
                lb = dim_info["lb"]
                inverse_map = dim_info["inverse_map"]
                start = static_len + tot_bin_len
                end = start + binary_len
                binary = new_x[start:end]
                int_value = lb.inverse_transform(np.asarray([binary]))[0]
                cat_value = inverse_map[int_value]
                categorical_new_x.append(cat_value)
                cat_index += 1
                tot_bin_len += binary_len
            else:
                categorical_new_x.append(new_x[i - cat_index])

        return categorical_new_x

    def _z_dims(self, all_xz_unsearched, all_xz_searched):
        """
        Prepare dims to use in preprocessing for categorical dimensions.
        Gathers a list of possible dimensions from stored and current z vectors.
        Not actually used for creating a list of possible search points, only
        for helping to convert possible search points from categorical to
        integer/float.

        Args:
            all_xz_unsearched ([list]): The collection of xz points which have
                not been searched.
            all_xz_searched ([list]): The collection of xz points which have
                been searched.

        Returns:
            ([tuple]) dimensions for the z space
        """

        all_z_unsearched = [
            split_xz(xz, self.x_dims, z_only=True) for xz in all_xz_unsearched
        ]
        all_z_searched = [
            split_xz(xz, self.x_dims, z_only=True) for xz in all_xz_searched
        ]
        all_z = all_z_searched + all_z_unsearched

        if not all_z:
            return []

        dims = [(z, z) for z in all_z[0]]

        for i, dim in enumerate(dims):
            cat_values = []
            for z in all_z:
                if type(z[i]) in dtypes.others:
                    # the dimension is categorical
                    if z[i] not in cat_values:
                        cat_values.append(z[i])
                        dims[i] = cat_values
        return dims
