from __future__ import unicode_literals, print_function, division

"""
The FireTask for running automatic optimization loops.

Please see the documentation for a comprehensive guide on usage. 
"""
import random
import heapq
import datetime
from itertools import product
from os import getpid, path
from time import sleep
from operator import mul
import warnings
from numpy import asarray
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, \
    ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from fireworks.utilities.fw_utilities import FW_BLOCK_FORMAT
from fireworks import FWAction, LaunchPad
from rocketsled.acq import acquire
from rocketsled.utils import deserialize, Dtypes
try:
    # for Python 3.6-
    import cPickle as pickle
except:
    # for Python 3.6+
    import pickle

try:
    # Python 3.6+ has no reduce() builtin
    from functools import reduce
except:
    pass


__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"


@explicit_serialize
class OptTask(FireTaskBase):
    """
    A FireTask for automatically running optimization loops and storing optimization data for complex workflows.

    OptTask takes in x and yi (input/output of current guess), gathers X (previous guesses input) and y (previous
    guesses output), and predicts the next best guess. 

    Required args:
        wf_creator (function): returns a workflow based on a unique vector, x.
        dimensions ([tuple]): each 2-tuple in the list defines one dimension in the search space in (low, high) format.
            For categorical dimensions, includes all possible categories as a list.
            Example: dimensions = dim = [(1,100), (9.293, 18.2838), ("red", "blue", "green")].
            
    Optional args:
    
        Database setup:
        host (string): The name of the MongoDB host where the optimization data will be stored.
        port (int): The number of the MongoDB port where the optimization data will be stored.
        name (string): The name of the MongoDB database where the optimization data will be stored.
        lpad (LaunchPad): A Fireworks LaunchPad object, which can be used to define the host/port/name of the db.
        opt_label (string): Names the collection of that the particular optimization's data will be stored in. Multiple
            collections correspond to multiple independent optimizations.
        db_extras (dict): Keyword arguments to be passed to MongoClient to help set up the db (e.g., password, username,
            SSL info)
            Example: db_extras={'username': 'myuser', 'password': 'mypassword', 'maxPoolSize': 10}
    
        Predictors:
        predictor (string): names a function which given a list of explored points and unexplored points, returns an 
            optimized guess. 
            Included sklearn predictors are:
                'RandomForestRegressor',
                'AdaBoostRegressor',
                'ExtraTreesRegressor',
                'GradientBoostingRegressor',
                'GaussianProcessRegressor',
                'LinearRegression',
                'SGDRegressor',
                'MLPRegressor',
                'KernelRidge',
                'SVR'
            Defaults to 'RandomForestRegressor'
            Example builtin predictor: predictor = 'SVR'
            Example custom predictor: predictor = 'my_module.my_predictor'
        predictor_args (list): the positional args to be passed to the model along with a list of points to be searched.
            For sklearn-based predictors included in OptTask, these positional args are passed to the init method of
            the chosen model. For custom predictors, these are passed to the chosen predictor function alongside the 
            searched guesses, the output from searched guesses, and an unsearched space to be used with optimization.
        predictor_kwargs (dict): the kwargs to be passed to the model. Similar to predictor_args.
        
        Predictor performance:
        n_search_points (int): The number of points to be searched in the search space when choosing the next best
            point. Choosing more points to search may increase the effectiveness of the optimization. The default is
            1000 points. 
        n_train_points (int): The number of already explored points to be chosen for training. Default is None, meaning
            all available points will be used for training. Reduce the number of points to decrease training times.
        space (str): The fully specified path of a pickle file containing a list of all possible searchable vectors.
            For example '/Users/myuser/myfolder/myspace.p'. When loaded, this space should be a list of tuples.
        acq (str): The acquisition function to use. Can be 'ei' for expected improvement, 'pi' for probability of improvement,
            or 'lcb' for lower confidence bound. Defaults to None, which means no acquisition function is used, and the highest
            predicted point is picked. Only applies to builtin predictors.
        n_bootstraps (int): The number of times each optimization should, sample, train, and predict values when generating
            uncertainty estimates for prediction. Only used if acq specified. At least 10 data points must be present for
            bootstrapping.

        Hyperparameter search:
        param_grid (dict): The sklearn-style dictionary to use for hyperparameter optimization. Each key should
            correspond to a regressor parameter name, and each value should be a list of possible settings for the
            parameter key. For example: param_grid={"n_estimators: [1, 10, 100], "max_features": ["sqrt", "auto", 3]}
        hyper_opt (int): Defines how hyperparamter search is performed. An int of 1 or greater defines the number of
            parameter combos to try with RandomizedSearchCV. A value of None means exhaustive search with GridSearchCV.
            param_grid must be defined to use this option. All searches are performed in parallel, if possible.

        Balancing exploration and exploitation
        random_interval (int): Suggests a random guess every n guesses instead of using the predictor suggestion. For
            instance, random_interval=10 has OptTask randomly guess every 1/10 predictions, and uses the predictor the
            other 9/10 times. Setting random_interval to an int greater than 1 may increase exploration. Default is
            None, meaning no random guesses.

        Extra features:
        get_z (string): the fully-qualified name of a function which, given a x vector, returns another vector z which
            provides extra information to the machine learner. The features defined in z are not used to run the
            workflow creator.
            Examples: 
                get_z = 'my_module.my_fun'
                get_z = '/path/to/folder/containing/my_package.my_module.my_fun'
        get_z_args (list): the positional arguments to be passed to the get_z function alongside x
        get_z_kwargs (dict): the kwargs to be passed to the get_z function alongside x
        persistent_z (str): The filename (pickle file) which should be used to store persistent z calculations. Specify
            this argument if calculating z for many (n_search_points) is not trivial and will cost time in computing.
            With this argument specified, each z will only be calculated once. Defaults to None, meaning that all
            unexplored z are calculated each iteration.
            Example:
                persistent_z = '/path/to/persistent_z_guesses.p'
        
        Miscellaneous:
        wf_creator_args (list): the positional args to be passed to the wf_creator function alongsize the new x vector
        wf_creator_kwargs (dict): details the kwargs to be passed to the wf_creator function alongside the new x vector
        encode_categorical (bool): If True, preprocesses categorical data (strings) to one-hot encoded binary arrays for
            use with custom predictor functions. Default False. 
        duplicate_check (bool): If True, checks that custom optimizers are not making duplicate guesses; all built-in
            optimizers cannot duplicate guess. If the custom predictor suggests a duplicate, OptTask picks a random
            guess out of the remaining untried space. Default is no duplicate check, and an error is raised if
            a duplicate is suggested.
        tolerances (list): The tolerance of each feature when duplicate checking. For categorical features, put 'None'
            Example: Our dimensions are [(1, 100), ['red', 'blue'], (2.0, 20.0)]. We want our first parameter to be
            a duplicate only if it is exact, and our third parameter to be a duplicate if it is within 1e-6. Then:
                tolerances=[0, None, 1e-6]
        max (bool): If true, makes optimization tend toward maximum values instead of minimum ones.
        batch_size (int): The number of jobs to submit per batch for a batch optimization. For example, batch_size=5
            will optimize every 5th job, then submitting another 5 jobs based on the best 5 predictions.
        timeout (int): The number of seconds to wait before resetting the lock on the db.

    Attributes:
        collection (MongoDB collection): The collection to store the optimization data.
        dtypes (Dtypes): Object containing the datatypes available for optimization.
        predictors ([str]): Built in sklearn regressors available for optimization with OptTask.
        launchpad (LaunchPad): The Fireworks LaunchPad object which determines where workflow data is stored.
        hyperopt (int): Defines the number of hyperparameter searches to be done, as per the FireTask argument.
        param_grid (dict): Defines the parameter grid for hyperparameter search, as per the FireTask argument.
        get_z (str): Fully qualified name of the "get_z" function defined by the user.
        _manager_query (dict/MongoDB query syntax): The document format which details how the manager (for parallel
            optimizations) are managed.
        _completed_query (dict/MongoDB query syntax): The document format which details how the optimization data (on 
            a per optimization loop basis) is stored for points which have already been computed/explored by a workflow. 
        _n_cats (int): The number of categorical dimensions.
        _encoding_info (dict): Data for converting between one-hot encoded data and categorical data.
    """
    _fw_name = "OptTask"
    required_params = ['wf_creator', 'dimensions']
    optional_params = ['host', 'port', 'name', 'lpad', 'opt_label', 'db_extras', 'predictor', 'predictor_args',
                       'predictor_kwargs', 'n_search_points', 'n_train_points', 'acq', 'random_interval', 'space', 'get_z',
                       'get_z_args', 'get_z_kwargs', 'wf_creator_args', 'wf_creator_kwargs', 'encode_categorical',
                       'duplicate_check', 'max', 'batch_size', 'tolerance', 'hyper_opt', 'param_grid', 'timeout',
                       'n_bootstraps']

    def run_task(self, fw_spec):
        """
        FireTask for running an optimization loop.

        Args:
            fw_spec (dict): the firetask spec. Must contain a '_y_opt' key with a float type field and must contain
                a '_x_opt' key containing a vector uniquely defining the search space.

        Returns:
            (FWAction) A workflow based on the workflow creator and a new, optimized guess. 
        """

        pid = getpid()
        sleeptime = .01
        timeout = self['timeout'] if 'timeout' in self else 180
        max_runs = int(timeout/sleeptime)
        max_resets = 3
        self._setup_db(fw_spec)

        # points for which a workflow has already been run
        self._completed_query = {'x': {'$exists': 1}, 'y': {'$exists': 1, '$ne': 'reserved'}, 'z': {'$exists': 1}}
        # the query format for the manager document
        self._manager_query = {'lock': {'$exists': 1}, 'queue': {'$exists': 1}}

        # Running stepwise optimization for concurrent processes requires a manual 'lock' on the optimization database
        # to prevent duplicate guesses. The first process sets up a manager document which handles locking and queueing
        # processes by PID. The single, active process in the lock is free to access optimization data; the queue of the
        # manager holds parallel process PIDs waiting to access the db. When the active process finishes, it removes
        # itself from the lock and moves the first queue PID into the lock, allowing the next process to begin
        # optimization. Each process continually tries to either queue or place itself into the lock if not active.

        for run in range(max_resets * max_runs):
            managers = self.collection.find(self._manager_query)

            if managers.count() == 0:
                self.collection.insert_one({'lock': pid, 'queue': []})
            elif managers.count() == 1:

                # avoid bootup problems if manager lock is being deleted concurrently with this check
                try:
                    manager = self.collection.find_one(self._manager_query)
                    manager_id = manager['_id']
                    lock = manager['lock']

                except TypeError:
                    continue

                if lock is None:
                    self.collection.find_one_and_update({'_id': manager_id}, {'$set': {'lock': pid}})

                elif lock != pid:
                    if pid not in manager['queue']:

                        # avoid bootup problems if manager queue is being deleted concurrently with this check
                        try:
                            self.collection.find_one_and_update({'_id': manager_id}, {'$push': {'queue': pid}})

                        except TypeError:
                            continue

                    else:
                        sleep(sleeptime)

                elif lock == pid:
                    try:
                        # required args
                        wf_creator = deserialize(self['wf_creator'])
                        x_dims = self['dimensions']

                        # predictor definition
                        predictor = 'RandomForestRegressor' if 'predictor' not in self else self['predictor']
                        pred_args = self['predictor_args'] if 'predictor_args' in self else []
                        pred_kwargs = self['predictor_kwargs'] if 'predictor_kwargs' in self else {}

                        # predictor performance
                        random_interval = self['random_interval'] if 'random_interval' in self else None
                        trainpts = self['n_train_points'] if 'n_train_points' in self else None
                        searchpts = self['n_search_points'] if 'n_search_points' in self else 1000
                        self.acq = self['acq'] if 'acq' in self else 'ei'
                        if self.acq not in [None, 'ei', 'pi', 'lcb']:
                            raise ValueError("Invalid acquisition function. Use 'ei', 'pi', 'lcb', or None.")
                        self.nstraps = self['n_bootstraps'] if 'n_bootstraps' in self else 10

                        # hyperparameter optimization
                        self.hyper_opt = self['hyper_opt'] if 'hyper_opt' in self else None
                        self.param_grid = self['param_grid'] if 'param_grid' in self else None
                        if self.hyper_opt and not self.param_grid:
                            raise ValueError("Please specify a param_grid.")

                        # extra features
                        self.get_z = deserialize(self['get_z']) if 'get_z' in self and self['get_z'] \
                                                                         is not None else lambda input_vector: []
                        get_z_args = self['get_z_args'] if 'get_z_args' in self else []
                        get_z_kwargs = self['get_z_kwargs'] if 'get_z_kwargs' in self else {}
                        persistent_z = self['persistent_z'] if 'persistent_z' in self else None

                        # miscellaneous
                        wf_creator_args = self['wf_creator_args'] if 'wf_creator_args' in self else []
                        wf_creator_kwargs = self['wf_creator_kwargs'] if 'wf_creator_kwargs' in self else {}
                        encode_categorical = self['encode_categorical'] if 'encode_categorical' in self else False
                        duplicate_check = self['duplicate_check'] if 'duplicate_check' in self else False
                        tolerances = self['tolerances'] if 'tolerances' in self else None
                        maximize = self['max'] if 'max' in self else False
                        batch_size = self['batch_size'] if 'batch_size' in self else 1

                        for kwargname, kwargdict in {'wf_creator_kwargs': wf_creator_kwargs,
                                               'get_z_kwargs': get_z_kwargs,
                                               'predictor_kwargs': pred_kwargs}.items():
                            if not isinstance(kwargdict, dict):
                                raise TypeError("{} should be a dictonary of keyword arguments.".format(kwargname))

                        for argname, arglist in {'wf_creator_args': wf_creator_args,
                                             'get_z_args': get_z_args,
                                             'predictor_args': pred_args}.items():
                            if not isinstance(arglist, list) or isinstance(arglist, tuple):
                                raise TypeError("{} should be a list/tuple of positional arguments".format(argname))

                        x = list(fw_spec['_x_opt'])
                        y = float(fw_spec['_y_opt'])

                        # If process A suggests a certain guess and runs it, process B may suggest the same guess while
                        # process A is running its new workflow. Therefore, process A must reserve the guess.
                        # Line below releases reservation on this document in case of workflow failure or end of workflow.
                        self.collection.delete_one({'x': x, 'y': 'reserved'})
                        self.dtypes = Dtypes()
                        self._check_dims(x_dims)

                        # fetch additional attributes for constructing machine learning model by calling get_z, if it exists
                        z = self.get_z(x, *get_z_args, **get_z_kwargs)

                        # use all possible training points as default
                        n_completed = self.collection.find(self._completed_query).count()
                        if not trainpts or trainpts > n_completed:
                            trainpts = n_completed

                        # check if an opimization should be done, when in batch mode
                        batch_mode = False if batch_size==1 else True
                        batch_ready = n_completed not in (0, 1) and (n_completed + 1) % batch_size == 0
                        if batch_mode and not batch_ready:

                            # 'None' predictor means this job was not used for an optimization run.
                            if self.collection.find_one({'x': x}):
                                if self.collection.find_one({'x': x, 'y': 'reserved'}):
                                    # For reserved guesses: update everything
                                    self.collection.find_one_and_update({'x': x, 'y': 'reserved'},
                                                                        {'$set': {'y': y, 'z': z, 'z_new': [], 'x_new': [],
                                                                                  'predictor': None,
                                                                                  'index': n_completed + 1}})

                                else:
                                    # For completed guesses (ie, this workflow is a forced duplicate), do not update index,
                                    # but update everything else
                                    self.collection.find_one_and_update({'x': x},
                                                                        {'$set': {'y': y, 'z': z, 'z_new': [], 'x_new': [],
                                                                                  'predictor': None}})
                            else:
                                # For new guesses: insert x, y, z, index, predictor, and dummy new guesses
                                self.collection.insert_one({'x': x, 'y': y, 'z': z, 'x_new': [], 'z_new': [],
                                                            'predictor': None, 'index': n_completed + 1})


                            self.pop_lock(manager_id)
                            return None

                        # Mongo aggregation framework may give duplicate documents, so we cannot use $sample to randomize
                        # the training points used
                        explored_indices = random.sample(range(1, n_completed + 1), trainpts)

                        Y = [y]
                        z = list(z)
                        XZ_explored = [x + z]
                        for i in explored_indices:
                            doc = self.collection.find_one({'index': i})
                            if doc is None:
                                raise ValueError("The doc with index {} does not exist".format(i))
                            XZ_explored.append(doc['x'] + doc['z'])
                            Y.append(doc['y'])

                        X_space = self._discretize_space(x_dims, discrete_floats=True)
                        X_space = list(X_space) if persistent_z else X_space

                        X_unexplored = []
                        for xi in X_space:
                            xj = list(xi)
                            if self.collection.find({'x': xj}).count() == 0 and xj != x:
                                X_unexplored.append(xj)
                                if len(X_unexplored) == searchpts:
                                    break

                        if persistent_z:
                            if path.exists(persistent_z):
                                with open(persistent_z, 'rb') as f:
                                    xz_map = pickle.load(f)
                            else:
                                xz_map = {tuple(xi): self.get_z(xi, *get_z_args, **get_z_kwargs) for xi in X_space}
                                with open(persistent_z, 'wb') as f:
                                    pickle.dump(xz_map, f)

                            XZ_unexplored = [xi + xz_map[tuple(xi)] for xi in X_unexplored]
                        else:
                            XZ_unexplored = [xi + self.get_z(xi, *get_z_args, **get_z_kwargs) for xi in X_unexplored]

                        # if there are no more unexplored points in the entire space, either they have been explored
                        # (ie have x, y, and z) or have been reserved.
                        if len(XZ_unexplored) < 1:
                            if self._is_discrete(x_dims, type='all'):
                                raise ExhaustedSpaceError("The discrete space has been searched exhaustively.")
                            else:
                                raise TypeError("A comprehensive list of points was exhausted but the dimensions are"
                                                "not discrete.")

                        xz_dims = x_dims + self._z_dims(XZ_unexplored, len(x))

                        # run machine learner on Z and X features
                        plist = [RandomForestRegressor, GaussianProcessRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor, LinearRegression, SGDRegressor, MLPRegressor, KernelRidge, SVR]
                        self.predictors = {p.__name__: p for p in plist}

                        if random_interval:
                            if random_interval < 1 or not isinstance(random_interval, int):
                                raise ValueError("The random interval must be an integer greater than 0.")
                            if n_completed % random_interval == 0:
                                predictor = 'random'

                        if predictor in self.predictors:
                            model = self.predictors[predictor]
                            XZ_explored = self._preprocess(XZ_explored, xz_dims)
                            XZ_unexplored = self._preprocess(XZ_unexplored, xz_dims)
                            scaling = False if self._is_discrete(dims=xz_dims, criteria='any') else True
                            XZ_onehot = self._predict(XZ_explored, Y, XZ_unexplored, model(*pred_args, **pred_kwargs), maximize, batch_size, scaling=scaling)
                            XZ_new = [self._postprocess(xz_onehot, xz_dims) for xz_onehot in XZ_onehot]

                        elif predictor == 'random':
                            XZ_new = random.sample(XZ_unexplored, batch_size)

                        else:
                            # If using a custom predictor, automatically convert categorical info to one-hot encoded ints
                            # Can be used when a custom predictor cannot use categorical info
                            if encode_categorical:
                                XZ_explored = self._preprocess(XZ_explored, xz_dims)
                                XZ_unexplored = self._preprocess(XZ_unexplored, xz_dims)

                            try:
                                predictor_fun = deserialize(predictor)
                            except Exception as E:
                                raise NameError("The custom predictor {} didnt import correctly!\n{}".format(predictor, E))

                            XZ_new = predictor_fun(XZ_explored, Y, XZ_unexplored, *pred_args, **pred_kwargs)
                            if not isinstance(XZ_new[0], list) or isinstance(XZ_new[0], tuple):
                                XZ_new = [XZ_new]

                        # duplicate checking for custom optimizer functions
                        if duplicate_check:

                            #todo: fix batch_mode duplicate checking
                            if batch_mode:
                                raise Exception("Dupicate checking in batch mode for custom predictors is not yet supported")

                            if predictor not in self.predictors and predictor != 'random':
                                X_new = [xz_new[:len(x)] for xz_new in XZ_new]
                                X_explored = [xz[:len(x)] for xz in XZ_explored]

                                if tolerances:
                                    for n, x_new in enumerate(X_new):
                                        if self._tolerance_check(x_new, X_explored, tolerances=tolerances):
                                            XZ_new[n] = random.choice(XZ_unexplored)

                                else:
                                    if self._is_discrete(x_dims):
                                        # test only for x, not xz because custom predicted z may not be accounted for
                                        for n, x_new in enumerate(X_new):
                                            if x_new in X_explored or x_new == x:
                                                XZ_new[n] = random.choice(XZ_unexplored)
                                    else:
                                        raise ValueError("Define tolerances parameter to duplicate check floats.")
                    except Exception:
                        self.pop_lock(manager_id)
                        raise

                    # make sure a process has not timed out and changed the lock pid while this process
                    # is computing the next guess
                    try:
                        if self.collection.find_one(self._manager_query)['lock'] != pid:
                            continue
                        else:
                            for xz_new in XZ_new:
                                # separate 'predicted' z features from the new x vector
                                x_new, z_new = xz_new[:len(x)], xz_new[len(x):]

                                # if it is a duplicate (such as a forced identical first guess)
                                forced_dupe = self.collection.find_one({'x': x})

                                acqmap = {"ei": "Expected Improvement", "pi": "Probability of Improvement", "lcb": "Lower Confidence Boundary", None: "Highest Value"}
                                predictorstr = predictor + " with acquisition: " + acqmap[self.acq]
                                if forced_dupe:
                                    # only update the fields which should be updated
                                    self.collection.find_one_and_update({'x': x},
                                                                        {'$set': {'y': y,
                                                                                  'z': z,
                                                                                  'z_new': z_new,
                                                                                  'x_new': x_new,
                                                                                  'predictor': predictorstr}})
                                    opt_id = forced_dupe['_id']
                                else:
                                    # update all the fields, as it is a new document
                                    res = self.collection.insert_one({'z': z,
                                                                      'y': y,
                                                                      'x': x,
                                                                      'z_new': z_new,
                                                                      'x_new': x_new,
                                                                      'predictor': predictorstr,
                                                                      'index': n_completed + 1})
                                    opt_id = res.inserted_id

                                # ensure previously fin. workflow results are not overwritten by concurrent predictions
                                if self.collection.find({'x': x_new,
                                                         'y': {'$exists': 1, '$ne': 'reserved'}}).count() == 0:
                                    # reserve the new x to prevent parallel processes from registering it as unexplored,
                                    # since the next iteration of this process will be exploring it
                                    self.collection.insert_one({'x': x_new, 'y': 'reserved'})
                                else:
                                    raise ValueError(
                                        "The predictor suggested a guess which has "
                                        "already been tried: {}".format(x_new))

                    except TypeError as E:
                        warnings.warn(
                            "Process {} timed out while computing next guess, with exception {}".format(pid, E),
                             RuntimeWarning)
                        continue

                    self.pop_lock(manager_id)

                    X_new = [xz_new[:len(x)] for xz_new in XZ_new]
                    new_wfs = [wf_creator(x_new, *wf_creator_args, **wf_creator_kwargs) for x_new in X_new]
                    for wf in new_wfs:
                        self.lpad.add_wf(wf)
                    return FWAction(update_spec={'_optimization_id': opt_id}, stored_data={'_optimization_id': opt_id})

            else:
                self.collection.delete_one(self._manager_query)

            if run in [max_runs*k for k in range(1, max_resets)]:
                self.collection.find_one_and_update(self._manager_query, {'$set': {'lock': None, 'queue': []}})

            elif run == max_runs*max_resets:
                raise StandardError("The manager is still stuck after resetting. Make sure no stalled processes are in"
                                    " the queue.")

    def _setup_db(self, fw_spec):
        """
        Sets up a MongoDB database for storing optimization data.

        Args:
            fw_spec (dict): The spec of the Firework which contains this Firetask.

        Returns:
            None
        """
        time_now = datetime.datetime.utcnow().strftime(FW_BLOCK_FORMAT)
        opt_label = self['opt_label'] if 'opt_label' in self else 'opt_default_' + time_now
        db_extras = self['db_extras'] if 'db_extras' in self else {}
        db_reqs = ('host', 'port', 'name')
        db_def = [req in self for req in db_reqs]

        if all(db_def):
            host, port, name = [self[k] for k in db_reqs]
            lpad = LaunchPad(host, port, name, **db_extras)

        elif any(db_def):
            raise AttributeError("Host, port, and name must all be specified!")

        elif 'lpad' in self:
            lpad_dict = self['lpad']
            lpad = LaunchPad.from_dict(lpad_dict)

        elif '_add_launchpad_and_fw_id' in fw_spec:
            if fw_spec['_add_launchpad_and_fw_id']:
                lpad = self.launchpad

        else:
            try:
                lpad = LaunchPad.auto_load()

            except AttributeError:
                # auto_load did not return any launchpad object, so nothing was defined.
                raise AttributeError("The optimization database must be specified explicitly (with host, port, and "
                                     "name), with Launchpad object (lpad), by setting _add_launchpad_and_fw_id to True "
                                     "in the fw_spec, or by defining LAUNCHPAD_LOC in your config file for "
                                     "LaunchPad.auto_load()")
        self.lpad = lpad
        self.collection = getattr(self.lpad.db, opt_label)

    def _check_dims(self, dims):
        """
        Ensure the dimensions are in the correct format for the optimization.
        
        Dimensions should be a list or tuple
        of lists or tuples each defining the search space in one dimension. The datatypes used inside each dimension's 
        definition should be NumPy compatible datatypes. Numerical dimensions (floats and ints) should take the form
        (upper, lower). Categorical dimensions should be an exhaustive list/tuple such as ['red', 'green', 'blue'].
        
        Args:
            dims (list): The dimensions of the search space. 

        Returns:
            None
        """
        dims_types = [list, tuple]

        if type(dims) not in dims_types:
            raise TypeError("The dimensions must be a list or tuple.")

        for dim in dims:
            if type(dim) not in dims_types:
                raise TypeError("The dimension {} must be a list or tuple.".format(dim))

            for entry in dim:
                if type(entry) not in self.dtypes.all:
                    raise TypeError("The entry {} in dimension {} cannot be used with OptTask."
                                    "A list of acceptable datatypes is {}".format(entry, dim, self.dtypes.all))

    def _is_discrete(self, dims, criteria='all'):
        """
        Checks if the search space is discrete.

        Args:
            dims ([tuple]): dimensions of the search space
            criteria (str/unicode): If 'all', returns bool based on whether ALL dimensions are discrete. If 'any', returns bool
                based on whether ANY dimensions are discrete.

        Returns:
            (bool) whether the search space is totally discrete.
        """

        if criteria=='all':
            for dim in dims:
                if type(dim[0]) not in self.dtypes.discrete or type(dim[1]) not in self.dtypes.discrete:
                    return False
            return True

        elif criteria=='any':
            for dim in dims:
                if type(dim[0]) in self.dtypes.discrete or type(dim[1]) in self.dtypes.discrete:
                    return True
            return False

    def _discretize_space(self, dims, discrete_floats=False, n_floats=100):
        """
        Create a list of points for searching during optimization. 

        Args:
            dims ([tuple]): dimensions of the search space. Individual dimensions should be in (higher, lower)
                form if integers, and should be a comprehensive list if categorical.
            discrete_floats (bool): If true, converts floating point (continuous) dimensions into discrete dimensions 
                by randomly sampling points between the dimension's (upper, lower) boundaries. If this is set to False, 
                and there is a continuous dimension, a list of all possible points in the space can't be calculated. 
            n_floats (int): Number of floating points to sample from continuous dimensions when discrete_float is True.

        Returns:
            ([list]) Points of the search space. 
        """

        if 'space' in self:
            if self['space']:
                with open(self['space'], 'rb') as f:
                    return pickle.load(f)

        total_dimspace = []

        for dim in dims:
            if len(dim) == 2:
                lower = dim[0]
                upper = dim[1]

                if type(lower) in self.dtypes.ints:
                    # Then the dimension is of the form (lower, upper)
                    dimspace = list(range(lower, upper + 1))
                elif type(lower) in self.dtypes.floats:
                    if discrete_floats:
                        dimspace = [random.uniform(lower, upper) for _ in range(n_floats)]
                    else:
                        raise ValueError("The dimension is a float. The dimension space is infinite.")
                else:  # The dimension is a discrete finite string list of two entries
                    dimspace = dim
            else:  # the dimension is a list of categories or discrete integer/float entries
                dimspace = dim

            random.shuffle(dimspace)
            total_dimspace.append(dimspace)
        space = [[xi] for xi in total_dimspace[0]] if len(dims) == 1 else product(*total_dimspace)

        return space

    def pop_lock(self, manager_id):
        """
        Releases the current process lock on the manager doc, and moves waiting processes from the queue to the lock.

        Args:
            manager_id: The MongoDB ObjectID object of the manager doc.

        Returns:
            None

        """
        queue = self.collection.find_one({'_id': manager_id})['queue']
        if not queue:
            self.collection.find_one_and_update({'_id': manager_id}, {'$set': {'lock': None}})
        else:
            new_lock = queue.pop(0)
            self.collection.find_one_and_update({'_id': manager_id},
                                                {'$set': {'lock': new_lock, 'queue': queue}})

    def _predict(self, X, Y, space, model, maximize, n_predictions, scaling=False):
        """
        Scikit-learn compatible model for stepwise optimization. It uses a regressive predictor evaluated on
        remaining points in a discrete space.
        
        Since sklearn modules cannot deal with categorical data, categorical data is preprocessed by _preprocess before
        being passed to _predict, and predicted x vectors are postprocessed by _postprocess to convert to the original 
        categorical dimensions.

        Args:
            X ([list]): List of vectors containing input training data.
            Y (list): List of scalars containing output training data.
            space ([list]): List of vectors containing all unexplored inputs. Should be preprocessed.
            model (sklearn model): The regressor used for predicting the next best guess.
            maximize (bool): Makes predictor return the guess which maximizes the predicted objective function output.
                Else minmizes the predicted objective function output.
            n_predictions (bool): Number of predictions to return (i.e. if 5, returns best 5 predictions)

        Returns:
            (list) A vector which is predicted to minimize (or maximize) the objective function.

        """

        # Scale data if all floats for dimensions
        if scaling:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            space = scaler.transform(space)

        if self.param_grid and len(X) > 10:
            predictor_name = model.__class__.__name__
            if predictor_name not in self.predictors:
                raise ValueError("Cannot perform automatic hyperparameter search with custom optimizer.")
            if not self.hyper_opt:
                n_combos = reduce(mul, [len(p) for p in list(self.param_grid.values())], 1)
                hp_selector = GridSearchCV(model, self.param_grid, n_jobs=n_combos)
            elif self.hyper_opt>=1:
                hp_selector = RandomizedSearchCV(model, self.param_grid, n_iter=self.hyper_opt, n_jobs=self.hyper_opt)
            else:
                raise ValueError("Automatic hyperparameter optimization must be either grid or random. Please set"
                                 "the hyper_opt parameter to None for GridSearchCV and to any integer larger than 1 for "
                                 "n iterations of RandomizedSearchCV.")
            hp_selector.fit(X, Y)
            model = model.__class__(**hp_selector.best_params_)

        if self.acq is None or len(X) < 10:
            model.fit(X, Y)
            values = model.predict(space).tolist()
            evaluator = heapq.nlargest if maximize else heapq.nsmallest
        else:
            # Use the acquistion function values
            values = acquire(self.acq, X, Y, space, model, maximize,
                             self.nstraps)
            evaluator = heapq.nlargest

        #todo: possible batch duplicates if two x predict the same y? .index() will find the first one twice
        predictions = evaluator(n_predictions, values)
        indices = [values.index(p) for p in predictions]
        return [space[i] for i in indices]

    def _preprocess(self, X, dims):
        """
        Transforms data containing categorical information to "one-hot" encoded data, since sklearn cannot process 
        categorical data on its own.
        
        Args:
            X ([list]): The search space, possibly containing categorical dimensions.
            dims: The dimensions of the search space. Used to define all possible choices for categorical dimensions
                so that categories are properly encoded.

        Returns:
            X ([list]): "One-hot" encoded forms of X data containing categorical dimensions. Search spaces which are 
                completely numerical are unchanged. 

        """
        self._n_cats = 0
        self._encoding_info = []

        for i, dim in enumerate(dims):
            if type(dim[0]) in self.dtypes.others:

                cats = [0] * len(X)
                for j, x in enumerate(X):
                    cats[j] = x[i - self._n_cats]

                forward_map = {k: v for v, k in enumerate(dim)}
                inverse_map = {v: k for k, v in forward_map.items()}

                lb = LabelBinarizer()
                lb.fit([forward_map[v] for v in dim])
                binary = lb.transform([forward_map[v] for v in cats])

                for j, x in enumerate(X):
                    del (x[i - self._n_cats])
                    x += list(binary[j])

                dim_info = {'lb': lb, 'inverse_map': inverse_map, 'binary_len': len(binary[0])}
                self._encoding_info.append(dim_info)
                self._n_cats += 1

        return X

    def _postprocess(self, new_x, dims):
        """
        Convert a "one-hot" encoded point (the predicted guess) back to the original categorical dimensions. 
        
        Args:
            new_x (list): The "one-hot" encoded new x vector predicted by the predictor.
            dims ([list]): The dimensions of the search space.

        Returns:
            categorical_new_x (list): The new_x vector in categorical dimensions. 

        """

        original_len = len(dims)
        static_len = original_len - self._n_cats
        categorical_new_x = []
        cat_index = 0
        tot_bin_len = 0

        for i, dim in enumerate(dims):
            if type(dim[0]) in self.dtypes.others:
                dim_info = self._encoding_info[cat_index]

                binary_len = dim_info['binary_len']
                lb = dim_info['lb']
                inverse_map = dim_info['inverse_map']

                start = static_len + tot_bin_len
                end = start + binary_len
                binary = new_x[start:end]

                int_value = lb.inverse_transform(asarray([binary]))[0]
                cat_value = inverse_map[int_value]
                categorical_new_x.append(cat_value)

                cat_index += 1
                tot_bin_len += binary_len

            else:
                categorical_new_x.append(new_x[i - cat_index])

        return categorical_new_x

    def _z_dims(self, XZ_unexplored, x_length):
        """
        Prepare dims to use in preprocessing for categorical dimensions. Gathers a list of possible dimensions from 
        stored and current z vectors. Not actually used for creating a list of possible search points, only for
        helping to convert possible search points from categorical to integer/float. 
        
        Returns:
            ([tuple]) dimensions for the z space
        """

        Z_unexplored = [z[x_length:] for z in XZ_unexplored]
        Z_explored = [doc['z'] for doc in self.collection.find(self._completed_query)]
        Z = Z_explored + Z_unexplored

        if not Z:
            return []

        dims = [(z, z) for z in Z[0]]

        for i, dim in enumerate(dims):
            cat_values = []
            for z in Z:
                if type(z[i]) in self.dtypes.others:
                    # the dimension is categorical
                    if z[i] not in cat_values:
                        cat_values.append(z[i])
                        dims[i] = cat_values
        return dims

    def _tolerance_check(self, x_new, X_explored, tolerances):
        """
        Duplicate checks with tolerances.

        Args:
            x_new: the new guess to be duplicate checked
            X_explored: the list of all explored guesses
            tolerances: the tolerances of each dimension

        Returns:
            True if x_new is a duplicate of a guess in X_explored.
            False if x_new is unique in the space and has yet to be tried.

        """

        if len(tolerances) != len(x_new):
            raise DimensionMismatchError("Make sure each dimension has a corresponding tolerance value of the same "
                                         "type! Your dimensions and the tolerances must be the same length and types."
                                         " Use 'None' for categorical dimensions.")

        # todo: there is a more efficient way to do this: abort check for a pair of points as soon as one dim...
        # todo: ...is outside of tolerance

        categorical_dimensions = []
        for i in range(len(x_new)):
            if type(x_new[i]) not in self.dtypes.numbers:
                categorical_dimensions.append(i)

        for x_ex in X_explored:
            numerical_dimensions_inside_tolerance = []
            categorical_dimensions_equal = []
            for i, _ in enumerate(x_new):
                if i in categorical_dimensions:
                    if str(x_new[i]) == str(x_ex[i]):
                        categorical_dimensions_equal.append(True)
                    else:
                        categorical_dimensions_equal.append(False)
                else:
                    if abs(float(x_new[i]) - float(x_ex[i])) <= float(tolerances[i]):
                        numerical_dimensions_inside_tolerance.append(True)
                    else:
                        numerical_dimensions_inside_tolerance.append(False)

            if all(numerical_dimensions_inside_tolerance) and all(categorical_dimensions_equal):
                return True

        # If none of the points inside X_explored are close to x_new (inside tolerance) in ALL dimensions, it is not a
        # duplicate
        return False


class ExhaustedSpaceError(Exception):
    pass

class DimensionMismatchError(Exception):
    pass
