"""
The FireTask for running automatic optimization loops.
"""
import sys
import random
import pickle
from itertools import product
from os import getpid
from time import sleep
from pymongo import MongoClient, ReturnDocument
from numpy import sctypes, asarray
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import LabelBinarizer
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from fireworks import FWAction, LaunchPad

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"

# TODO: @ardunn - In retrain_interval, why use a random guess? That's nuts. The idea here was to use the previously trained model to guess (say) the second best solution, the third best solution, etc. If this is too complicated, then I would just remove the parameter. There is already n_search_points and n_train_points to make things go faster if needed. You could repurpose this parameter to be something just for doing a random guess every once in a while (e.g., once every 10 iterations) to help balance exploration and exploitation. - AJ
# TODO: @ardunn - The default for n_train_points should be None, which would signify to use all the already-explored points to build the model. It is clunky to have to set this to some arbitrary large number. - AJ
# TODO: @ardunn - docs for n_generation points is unclear. - AJ
# TODO: @ardunn - there is just a lot of parameters documented here. Suggest some reorganization (e.g., put related parameters next to one another) and also suggest more comprehensive docs outside the code, i.e., in the official Turboworks docs.

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
        get_z (string): the fully-qualified name of a function which, given a x vector, returns another vector z which
            provides extra information to the machine learner. The features defined in z are not used to run the
            workflow creator.
            Examples: 
                get_z = 'my_module.my_fun'
                get_z = '/path/to/folder/containing/my_package.my_module.my_fun'
        predictor (string): names a function which given a list of explored points and unexplored points, returns an 
            optimized guess. 
            Included sklearn predictors are:
                'LinearRegression'
                'RandomForestRegressor'
                'AdaBoostRegressor'
                'BaggingRegressor'
                'GradientBoostingRegressor'
                'GaussianProcessRegressor'
                'MLPRegressor'
                'SVR'
            Defaults to 'RandomForestRegressor'
            Example builtin predictor: predictor = 'SVR'
            Example custom predictor: predictor = 'my_module.my_predictor'
        max (bool): If true, makes optimization tend toward maximum values instead of minimum ones.
        wf_creator_args (list): the positional args to be passed to the wf_creator function alongsize the new x vector
        wf_creator_kwargs (dict): details the kwargs to be passed to the wf_creator function alongside the new x vector
        duplicate_check (bool): If True, checks for duplicate guess in discrete, finite spaces. Default is no
            duplicate check.
        host (string): The name of the MongoDB host where the optimization data will be stored.
        port (int): The number of the MongoDB port where the optimization data will be stored.
        name (string): The name of the MongoDB database where the optimization data will be stored.
        lpad (LaunchPad): A Fireworks LaunchPad object.
        opt_label (string): Names the collection of that the particular optimization's data will be stored in. Multiple
            collections correspond to multiple independent optimizations.
        retrain_interval (int): The number of iterations to wait before retraining the expensive model. On iterations
            where the model is not trained, a random guess is used. 
        n_search_points (int): The number of points to be searched in the search space when choosing the next best
            point. Choosing more points to search may increase the effectiveness of the optimization. The default is
            1000 points. 
        n_train_points (int): The number of already explored points to be chosen for training. Default is None, meaning
            all available points will be used for training. Reduce the number of points to decrease training times.
        n_generation_points (int): The number of points to be generated and stored the database during creation of the
            search space. Use this if your search space is gigantic, otherwise OptTask will try to store a huge 
            search space in your database!
        space (str): The fully specified path of a pickle file containing a list of all possible searchable vectors.
            For example '/Users/myuser/myfolder/myspace.p'. When loaded, this space should be a list of tuples. 
        predictor_args (list): the positional args to be passed to the model along with a list of points to be searched.
            For sklearn-based predictors included in OptTask, these positional args are passed to the init method of
            the chosen model. For custom predictors, these are passed to the chosen predictor function alongside the 
            searched guesses, the output from searched guesses, and an unsearched space to be used with optimization.
        predictor_kwargs (dict): the kwargs to be passed to the model. Similar to predictor_args.
        get_z_args (list): the positional arguments to be passed to the get_z function alongside x
        get_z_kwargs (dict): the kwargs to be passed to the get_z function alongside x
        encode_categorical (bool): If True, preprocesses categorical data (strings) to one-hot encoded binary arrays for
            use with custom predictor functions. Default False. 
            
    Attributes:
        collection (MongoDB collection): The collection to store the optimization data.
        dtypes (Dtypes): Object containing the datatypes available for optimization.
        predictors ([str]): Built in sklearn regressors available for optimization with OptTask.
        launchpad (LaunchPad): The Fireworks LaunchPad object which determines where workflow data is stored.
        _manager_query (dict/MongoDB query syntax): The document format which details how the manager (for parallel
            optimizations) are managed.
        _explored_query (dict/MongoDB query syntax): The document format which details how the optimization data (on 
            a per optimization loop basis) is stored for explored points. 
        _unexplored_inclusive_query (dict/MongoDB query syntax): The document format which details how optimization
            data is stored for unexplored points including the current point.
        _unexplored_noninclusive_query (dict/MongoDB query syntax): Similar to unexplored_inclusive_format but not
            including the current guess.
        _n_cats (int): The number of categorical dimensions.
        _encoding_info (dict): Data for converting between one-hot encoded data and categorical data.
    """
    _fw_name = "OptTask"
    required_params = ['wf_creator', 'dimensions']
    optional_params = ['get_z', 'predictor', 'max', 'wf_creator_args', 'wf_creator_kwargs', 'duplicate_check',
                       'host', 'port', 'name', 'lpad', 'opt_label', 'retrain_interval', 'n_train_points',
                       'n_search_points', 'n_generation_points', 'space', 'predictor_args', 'predictor_kwargs',
                       'get_z_args', 'get_z_kwargs', 'encode_categorical']

    def run_task(self, fw_spec):
        """
        FireTask for running an optimization loop.

        Args:
            fw_spec (dict): the firetask spec. Must contain a '_y_opt' key with a float type field and must contain
                a '_x_opt' key containing a vector uniquely defining the search space.

        Returns:
            (FWAction) A workflow based on the workflow creator and a new, optimized guess. 
        """

        # the pid identifies the process during parallel duplicate checking
        pid = getpid()
        sleeptime = .01
        max_runs = 1000
        max_resets = 10
        self._setup_db(fw_spec)
        self._set_queries(fw_spec)

        # Running stepwise optimization for concurrent processes requires a manual 'lock' on the optimization database
        # to prevent duplicate guesses. The first process sets up a manager document which handles locking and queuing
        # processes by PID. The single, active process in the lock is free to access optimization data; the queue of the
        # manager holds parallel process PIDs waiting to access the db. When the active process finishes, it removes
        # itself from the lock and moves the first queue PID into the lock, allowing the next process to begin
        # optimization. Each process continually tries to either queue itself of place itself into the lock if not
        # already active.

        for run in range(max_resets * max_runs):
            managers = self.collection.find(self._manager_query)

            if managers.count() == 0:
                self.collection.insert_one({'lock': pid, 'queue': []})
            elif managers.count() == 1:

                # avoid bootup problems if manager docs are being deleted concurrently with this check
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

                        # avoid bootup problems if manager docs are being deleted concurrently with this check
                        try:
                            self.collection.find_one_and_update({'_id': manager_id}, {'$push': {'queue': pid}})

                        except TypeError:
                            continue

                    else:
                        sleep(sleeptime)

                elif lock == pid:

                    x = fw_spec['_x_opt']
                    y = fw_spec['_y_opt']

                    # If process A suggests a certain guess and runs it, process B may suggest the same guess while
                    # process A is running its new workflow. Therefore, process A must reserve the guess.
                    # Line below releases reservation on this document in case of workflow failure or end of workflow.
                    self.collection.find_one_and_update({'x': x}, {'$set': {'y': y}})
                    self.dtypes = Dtypes()

                    x_dims = self['dimensions']
                    self._check_dims(x_dims)

                    # fetch additional attributes for constructing machine learning model by calling get_z, if it exists
                    self.get_z = self._deserialize(self['get_z']) if 'get_z' in self else lambda input_vector: []
                    get_z_args = self['get_z_args'] if 'get_z_args' in self else []
                    get_z_kwargs = self['get_z_kwargs'] if 'get_z_kwargs' in self else {}
                    z = self.get_z(x, *get_z_args, **get_z_kwargs)

                    train_points = self['n_train_points'] if 'n_train_points' in self else 1000  # TODO: @ardunn - put these variable definitions (including default values) at top of function. Same with other similar params. - AJ
                    search_points = self['n_search_points'] if 'n_search_points' in self else 1000
                    explored_docs = self.collection.find(self._explored_query, limit=train_points)  # TODO: @ardunn - ideally you want a random selection for the limit. Look up how to do this in Mongo so you don't always get the same records back. - AJ
                    persistent_search = self['persistent_search'] if 'persistent_search' in self else False

                    Y = [y]
                    XZ_explored = [x + z]
                    for doc in explored_docs:
                        XZ_explored.append(doc['x'] + doc['z'])
                        Y.append(doc['y'])

                    if persistent_search:
                        unexplored_docs = self.collection.find(self._unexplored_noninclusive_query, limit=search_points)  # TODO: @ardunn - ideally you want a random selection for the limit. Look up how to do this in Mongo so you don't always get the same records back. - AJ

                        # if no comprehensive list has been made, insert some unexplored docs
                        if unexplored_docs.count() == 0:

                            X_space = self._discretize_space(x_dims, discrete_floats=True)
                
                            # prevent a huge discrete space from being stored initially in db (can be very slow!)
                            generation_points = self['n_generation_points'] if 'n_generation_points' in self else 20000  # TODO: @ardunn - is "n_generation_points" the amount of unexplored points that are generated at one time (but we may run the generation multiple times, or is it the maximum number of points that can possibly be explored by TurboWorks?  - AJ
                            stored_docs = 0

                            for xi in X_space:
                                xj = list(xi)
                                if self.collection.find({'x': xj}).count() == 0 and xj != x:
                                    if stored_docs < generation_points:
                                        self._store({'x': xj, 'z': self.get_z(xj, *get_z_args, **get_z_kwargs)})
                                        stored_docs += 1
                                    else:
                                        break

                            unexplored_docs = self.collection.find(self._unexplored_inclusive_query, limit=search_points)

                        # append the additional attributes to the unique vectors to feed into machine learning
                        XZ_unexplored = [doc['x'] + doc['z'] for doc in unexplored_docs]

                    else:
                        X_space = self._discretize_space(x_dims, discrete_floats=True)
                        X_unexplored = []
                        for xi in X_space:
                            xj = list(xi)
                            if self.collection.find({'x': xj, 'y': {'$exists': 1}}).count() == 0 and xj != x:
                                X_unexplored.append(xj)
                                
                                if len(X_unexplored) == search_points:
                                    break
                        
                        XZ_unexplored = [xi + self.get_z(xi) for xi in X_unexplored]

                    # there are no more unexplored points in the entire space
                    if len(XZ_unexplored) <= 1:
                        if self._is_discrete(x_dims):
                            raise Exception("The discrete space has been searched exhaustively.")
                        else:
                            raise TypeError("A comprehensive list of points was exhausted but the dimensions are"
                                            "not discrete.")

                    xz_dims = x_dims + self._z_dims(XZ_unexplored, len(x))

                    # run machine learner on Z and X features
                    random_interval = self['random_interval'] if 'random_interval' in self else None
                    encode_categorical = self['encode_categorical'] if 'encode_categorical' in self else False

                    self.predictors = ['RandomForestRegressor',
                                       'AdaBoostRegressor',
                                       'BaggingRegressor',
                                       'GradientBoostingRegressor',
                                       'GaussianProcessRegressor',
                                       'LinearRegression',
                                       'MLPRegressor',
                                       'SVR']

                    predictor = 'RandomForestRegressor' if 'predictor' not in self else self['predictor']

                    if random_interval:
                        if self.collection.find(self._explored_query).count() % random_interval == 0:
                            predictor = 'random_guess'

                    pred_args = self['predictor_args'] if 'predictor_args' in self else []
                    pred_kwargs = self['predictor_kwargs'] if 'predictor_kwargs' in self else {}

                    if predictor in self.predictors:

                        if predictor == 'RandomForestRegressor':
                            model = RandomForestRegressor
                        elif predictor == 'AdaBoostRegressor':
                            model = AdaBoostRegressor
                        elif predictor == 'BaggingRegressor':
                            model = BaggingRegressor
                        elif predictor == 'GradientBoostingRegressor':
                            model = GradientBoostingRegressor
                        elif predictor == 'GaussianProcessRegressor':
                            model = GaussianProcessRegressor
                        elif predictor == 'LinearRegression':
                            model = LinearRegression
                        elif predictor == 'MLPRegressor':
                            model = MLPRegressor
                        elif predictor == 'SVR':
                            model = SVR
                        else:
                            raise NameError("{} was in the predictor list but did not have a model!".format(predictor))

                        maximize = self['max'] if 'max' in self else False
                        XZ_explored = self._preprocess(XZ_explored, xz_dims)
                        XZ_unexplored = self._preprocess(XZ_unexplored, xz_dims)
                        xz_onehot = self._predict(XZ_explored, Y, XZ_unexplored, model(*pred_args, **pred_kwargs), maximize)
                        xz_new = self._postprocess(xz_onehot, xz_dims)

                    elif predictor == 'random_guess':
                        xz_new = random.choice(XZ_unexplored)

                    else:
                        # If using a custom predictor, automatically convert categorical info to one-hot encoded ints
                        # Can be used when a custom predictor cannot use categorical info
                        if encode_categorical: 
                            XZ_explored = self._preprocess(XZ_explored, xz_dims)
                            XZ_unexplored = self._preprocess(XZ_unexplored, xz_dims)

                        try:
                            predictor_fun = self._deserialize(predictor)

                        except ImportError as E:
                            raise NameError("The custom predictor {} didnt import correctly!\n{}".format(predictor, E))

                        xz_new = predictor_fun(XZ_explored, Y, XZ_unexplored, *pred_args, **pred_kwargs)

                    # duplicate checking for custom optimizer functions
                    if 'duplicate_check' in self and predictor not in self.predictors and predictor != 'random_guess':
                        if self['duplicate_check']:
                            if self._is_discrete(x_dims):
                                x_new = xz_new[:len(x)]
                                X_explored = [xz[:len(x)] for xz in XZ_explored]
                                # test only for x, not xz because custom predicted z may not be accounted for
                                if x_new in X_explored:
                                    xz_new = random.choice(XZ_unexplored)

                    # separate 'predicted' z features from the new x vector
                    x_new, z_new = xz_new[:len(x)], xz_new[len(x):]

                    # make sure a process has not timed out and changed the lock pid while this process
                    # is computing the next guess
                    try:
                        if self.collection.find_one(self._manager_query)['lock'] != pid:
                            continue
                        else:
                            opt_id = self._store({'z': z, 'y': y, 'x': x, 'z_new': z_new, 'x_new': x_new})

                            # reserve the new x prevent to prevent parallel processes from registering it as unexplored
                            # since the next iteration of this process will be exploring it
                            self.collection.find_one_and_update({'x': x_new}, {'$set': {'y': []}})  # TODO: @ardunn - confirm that some other process did not run the desired x and have an actual y in the database. Otherwise you might overwrite an actual value with nothing. - AJ

                    except TypeError:
                        continue

                    queue = self.collection.find_one({'_id': manager_id})['queue']
                    if not queue:
                        self.collection.find_one_and_update({'_id': manager_id}, {'$set': {'lock': None}})
                    else:
                        new_lock = queue.pop(0)
                        self.collection.find_one_and_update({'_id': manager_id},
                                                            {'$set': {'lock': new_lock, 'queue': queue}})

                    wf_creator = self._deserialize(self['wf_creator'])

                    wf_creator_args = self['wf_creator_args'] if 'wf_creator_args' in self else []
                    if not isinstance(wf_creator_args, list) or isinstance(wf_creator_args, tuple):
                        raise TypeError("wf_creator_args should be a list/tuple of positional arguments")

                    wf_creator_kwargs = self['wf_creator_kwargs'] if 'wf_creator_kwargs' in self else {}
                    if not isinstance(wf_creator_kwargs, dict):
                        raise TypeError("wf_creator_kwargs should be a dictonary of keyword arguments.")

                    return FWAction(additions=wf_creator(x_new, *wf_creator_args, **wf_creator_kwargs),
                                    update_spec={'optimization_id': opt_id})

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

        # TODO: @ardunn - doesn't look like this process will work with password-protected LaunchPad. Most people have their FWS databases password-protected.  - AJ

        opt_label = self['opt_label'] if 'opt_label' in self else 'opt_default'
        db_reqs = ('host', 'port', 'name')
        db_defined = [req in self for req in db_reqs]

        if all(db_defined):
            host, port, name = [self[k] for k in db_reqs]

        elif any(db_defined):
            raise AttributeError("Host, port, and name must all be specified!")

        elif 'lpad' in self:
            lpad = self['lpad']
            host, port, name = [lpad[req] for req in db_reqs]

        elif '_add_launchpad_and_fw_id' in fw_spec:
            if fw_spec['_add_launchpad_and_fw_id']:
                host, port, name = [getattr(self.launchpad, req) for req in db_reqs]

        else:
            try:
                host, port, name = [getattr(LaunchPad.auto_load(), req) for req in db_reqs]

            except AttributeError:
                # auto_load did not return any launchpad object, so nothing was defined.
                raise AttributeError("The optimization database must be specified explicitly (with host, port, and "
                                     "name), with Launchpad object (lpad), by setting _add_launchpad_and_fw_id to True "
                                     "in the fw_spec, or by defining LAUNCHPAD_LOC in fw_config.py for "
                                     "LaunchPad.auto_load()")  # TODO: @ardunn - LAUNCHPAD_LOC is typically not set through fw_config.py (that requires modifying FWS source code), it's set through a config file: https://hackingmaterials.lbl.gov/fireworks/config_tutorial.html  - AJ

        mongo = MongoClient(host, port)
        db = getattr(mongo, name)
        self.collection = getattr(db, opt_label)

    def _set_queries(self, fw_spec):
        """
        Sets the formats which the optimization database will use to query data. The class attributes here are used to 
        distinguish between points already explored (ie a workflow has been run using this point) and points which have
        yet to be explored (ie, for which a workflow has not been run). The query format of the manager is also set here.
            
        Args:
            fw_spec (dict): The spec of the Firework which contains this Firetask.

        Returns:
            None

        """
        x = fw_spec['_x_opt']

        # points for which a workflow has already been run
        self._explored_query = {'x': {'$exists': 1}, 'yi': {'$ne': [], '$exists': 1}, 'z': {'$exists': 1}}

        # points for which a workflow has not already been run, including the current point (already run)
        self._unexplored_inclusive_query = {'x': {'$exists': 1}, 'yi': {'$exists': 0}}

        # points for which a workflow has not already been run, not including the current point
        self._unexplored_noninclusive_query = {'x': {'$ne': x, '$exists': 1}, 'yi': {'$exists': 0}}

        # the query format for the manager document
        self._manager_query = {'lock': {'$exists': 1}, 'queue': {'$exists': 1}}

    def _check_dims(self, dims):
        """
        Ensure the dimensions are in the correct format for the optimization.
        
        Dimensions should be a list or tuple
        of lists or tuples each defining the search space in one dimension. The datatypes used inside each dimension's 
        definition should be NumPy compatible datatypes. Numerical dimensions (floats and ints) should take the form (upper, lower). Categorical dimensions should be 
        an exhaustive list/tuple such as ['red', 'green', 'blue'].
        
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

    def _store(self, spec):
        """
        Stores and updates turboworks database files and prevents parallel initial guesses. 

        Args:
            spec (dict): a turboworks-generated spec (or subset of a spec) to be stored in the turboworks db.

        Returns:
            (ObjectId) the PyMongo BSON id object for the document inserted/updated.
        """
        new_doc = self.collection.find_one_and_replace({'x': spec['x']},
                                                       spec,
                                                       upsert=True,
                                                       return_document=ReturnDocument.AFTER)
        return new_doc['_id']

    def _deserialize(self, fun):
        """
        Takes a fireworks serialzed function handle and maps it to a function object.

        Args:
            fun (string): a 'module.function' or '/path/to/mod.func' style string specifying the function

        Returns:
            (function) The function object defined by fun
        """
        # todo: merge with PyTask's deserialize code, move to fw utils

        toks = fun.rsplit(".", 1)
        modname, funcname = toks

        if "/" in toks[0]:
            path, modname = toks[0].rsplit("/", 1)
            sys.path.append(path)

        mod = __import__(str(modname), globals(), locals(), fromlist=[str(funcname)])
        return getattr(mod, funcname)

    def _is_discrete(self, dims):
        """
        Checks if the search space is totally discrete.

        Args:
            dims ([tuple]): dimensions of the search space

        Returns:
            (bool) whether the search space is totally discrete.
        """

        for dim in dims:
            if type(dim[0]) not in self.dtypes.discrete or type(dim[1]) not in self.dtypes.discrete:
                return False
        return True

    def _discretize_space(self, dims, discrete_floats=False, n_floats=100):
        """
        Create a list of points for searching during optimization. 

        Args:
            dims ([tuple]): dimensions of the search space. Individual dimensions should be in (higher, lower)
                form if integers, and should be a comprehensive list if categorical.
            n_points (int): number of points to search. If None, uses all possible points in the search space.
            discrete_floats (bool): If true, converts floating point (continuous) dimensions into discrete dimensions 
                by randomly sampling points between the dimension's (upper, lower) boundaries. If this is set to False, 
                and there is a continuous dimension, a list of all possible points in the space can't be calculated. 
            n_floats (int): Number of floating points to sample from continuous dimensions when discrete_float is True.

        Returns:
            ([list]) Points of the search space. 
        """

        if 'space' in self:
            if self['space']:
                return pickle.load(open(self['space'], 'rb'))

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
                        dimspace = [random.uniform(lower, upper) for i in range(n_floats)]
                    else:
                        raise ValueError("The dimension is a float. The dimension space is infinite.")
                else:  # The dimension is a discrete finite string list of two entries
                    dimspace = dim
            else: # the dimension is a list of categories or discrete integer/float entries
                dimspace = dim

            random.shuffle(dimspace)
            total_dimspace.append(dimspace)
        space = [[xi] for xi in total_dimspace[0]] if len(dims) == 1 else product(*total_dimspace)

        return space

    def _predict(self, X, y, space, model, maximize):
        """
        Scikit-learn compatible model for stepwise optimization. It uses a regressive predictor evaluated on
        remaining points in a discrete space.
        
        Since sklearn modules cannot deal with categorical data, categorical data is preprocessed by _preprocess before
        being passed to _predict, and predicted x vectors are postprocessed by _postprocess to convert to the original 
        categorical dimensions.

        Args:
            X ([list]): List of vectors containing input training data.
            y (list): List of scalars containing output training data.
            space ([list]): List of vectors containing all possible inputs. Should be preprocessed before being 
            passed to
                predictor function.
            model (sklearn model): The regressor used for predicting the next best guess.
            maximize (bool): Makes predictor return the guess which maximizes the predicted objective function output.
                Else minmizes the predicted objective function output.  

        Returns:
            (list) A vector which is predicted to minimize (or maximize) the objective function.

        """
        model.fit(X, y)
        values = model.predict(space).tolist()
        evaluator = max if maximize else min
        min_or_max = evaluator(values)
        i = values.index(min_or_max)
        return space[i]

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
        Z_explored = [doc['z'] for doc in self.collection.find(self._explored_query)]
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


class Dtypes(object):
    """
    Defines the datatypes available for optimization.
    """

    def __init__(self):
        d = sctypes
        self.ints = d['int'] + d['uint'] + [int]
        self.floats = d['float'] + [float]
        self.reals = self.ints + self.floats
        self.complex = d['complex']
        self.numbers = self.reals + self.complex
        self.others = d['others']
        self.discrete = self.ints + self.others
        self.all = self.numbers + self.others

def random_guess(dimensions, dtypes=Dtypes()):
    """
    Returns random new inputs based on the dimensions of the search space.
    It works with float, integer, and categorical types

    Args:
        dimensions ([tuple]): defines the dimensions of each parameter
            example: [(1,50),(-18.939,22.435),["red", "green" , "blue", "orange"]]

    Returns:
        random_vector (list): randomly chosen next parameters in the search space
            example: [12, 1.9383, "green"]
    """

    random_vector = []

    for dimset in dimensions:
        upper = dimset[1]
        lower = dimset[0]
        if type(lower) in dtypes.ints:
            new_param = random.randint(lower, upper)
            random_vector.append(new_param)
        elif type(lower) in dtypes.floats:
            new_param = random.uniform(lower, upper)
            random_vector.append(new_param)
        elif type(lower) in dtypes.others:
            domain_size = len(dimset)-1
            new_param = random.randint(0, domain_size)
            random_vector.append(dimset[new_param])
        else:
            raise TypeError("The type {} is not supported by dummy opt as a categorical or "
                            "numerical type".format(type(upper)))

    return random_vector
