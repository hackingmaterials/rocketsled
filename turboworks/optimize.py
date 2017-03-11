"""
The main FireTasks for running automatic optimization loops are contained in this module.
"""

import sys
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from fireworks import FWAction
from pymongo import MongoClient
from turboworks.references import dtypes
from turboworks.discrete import calculate_discrete_space


__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"


@explicit_serialize
class OptTask(FireTaskBase):
    """
    A FireTask for running optimization loops automatically.


    :param wf_creator: (function) returns a workflow based on the input of a unique vector, z.

    :param dimensions: (list of tuples) defines the search space in (low, high) format. For categorical dimensions,
    includes all possible categories.
        example: dimensions = [(1,100), (9.293, 18.2838) ("red, "blue", "green")].

    :param get_x: (string) names a function which, given a z vector, returns another vector x which provides extra
    information to the machine learner. The features defined in x are not used to run the workflow creator.
        example: get_x = 'my_module.my_fun'

    Strings including path names to directory folders can also be used. Simply append the module name and function name
    in python "." format (e.g., mod.func) to the end of the path name (e.g., /Users/me/Documents/project).
        example: get_x = '/path/to/module/mod.func'

    :param predictor: (string) names a function which given a list of inputs, a list of outputs, and a dimensions space,
     can return a new optimized input vector. Can specify either a skopt function or a custom function.
        example: predictor = 'my_module.my_predictor'

    :param wf_creator_args: (dict) details the kwargs to be passed to the wf_creator function alongside the z vector
        example:
        If the wf_creator function is declared as
            wf_creator(z, param1=14, param2="glass")
        The wf_creator_args should be
            {'param1':14, 'param2':'glass'}

    :param duplicate_check: (boolean) If True, checks for duplicate guesss in discrete, finite spaces. (NOT currently
    working with concurrent workflows). Default is no duplicate check.
    """

    _fw_name = "OptTask"
    required_params = ['wf_creator', 'dimensions']
    optional_params = ['get_x', 'predictor', 'wf_creator_args', 'duplicate_check']

    def __init__(self, *args, **kwargs):

        """
        Initialization of OptTask object.

        :param args: variable positional args for initialization
        :param kwargs: variable keyword args for initialization
        """

        # todo: cleanup or refactor

        super(FireTaskBase, self).__init__(*args, **kwargs)

        self._tw_port = 27017
        self._tw_host = 'localhost'
        self._tw_mongo = MongoClient(self._tw_host, self._tw_port)
        self._tw_db = self._tw_mongo.turboworks
        self._tw_collection = self._tw_db.turboworks
        self.optimizers = ['gbrt_minimize', 'dummy_minimize', 'forest_minimize', 'gp_minimize']

    def _store(self, spec, update = False, id = None):
        """
        Stores and updates turboworks database files.

        :param spec: (dict) a turboworks-generated spec (or sub-spec) to be stored in the turboworks db.
        :param update: (boolean) whether to update the document (True) or insert a new one (False)
        :param id: (ObjectId BSON object) the mongodb id object. if update == True, updates the document with this id.
        :return: (ObjectId BSON object) the mongodb id object for the document inserted/updated.
        """

        if update == False:
            return self._tw_collection.insert_one(spec)
        else:
            return self._tw_collection.update({"_id":id },{'$set' : spec})

    def _deserialize_function(self, fun):
        """
        Takes a fireworks serialzed function handle and maps it to a function object.

        :param fun: (String) a 'module.function' or '/path/to/mod.func' style string specifying the function
        :return: (function) the function object defined by fun
        """

        #todo: merge with PyTask's deserialze code, move to fw utils

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

        :param dims: (list of tuples) dimensions of the search space
        :return: (boolean) whether the search space is totally discrete.
        """

        for dim in dims:
            if type(dim[0]) not in dtypes.discrete or type(dim[1]) not in dtypes.discrete:
                return False
        return True

    def _dupe_check(self, z, Z_dim):

        available_z = calculate_discrete_space(Z_dim)

        for doc in self._collection:
            if tuple(doc['z']) in available_z:
                available_z.remove(tuple(doc['z']))

        if len(available_z) == 0:
            raise ValueError("The search space has been exhausted.")

        if z in available_z:
            return z
        else:
            import random
            return random.choice(available_z)

    @property
    def _collection(self):
        """
        Wrapper of .find() pymongo method for easy access to most up to date _collection.

        :return: (PyMongo cursor object) the results of an empty turboworks database query.
        """

        return self._tw_collection.find()

    @property
    def _X_dims(self):
        """
        Creates some X dimensions so that the optimizer can run without the user specifing the X dimension range.
        Simply sets each dimension equal to the (lowest, highest) values of any x for that dimension in the database.

        If there is only one document in the database, it sets the dimension to slightly higher and lower values than
        the x dimension value.

        For categorical dimensions, it includes all dimensions in X.

        :return: (list of tuples) a list of dimensions
        """

        X = [doc['x'] for doc in self._collection]
        dims = [[x, x] for x in X[0]]
        check = dims

        cat_values = []

        for x in X:
            for i, dim in enumerate(dims):
                if type(x[i]) in dtypes.others:
                    # the dimension is categorical
                    if x[i] not in cat_values:
                        cat_values.append(x[i])
                        dims[i] = cat_values
                else:
                    if x[i] < dim[0]:
                        # this value is the new minimum
                        dims[i][0] = x[i]
                    elif x[i] > dim[1]:
                        # this value is the new maximum
                        dims[i][1] = x[i]
                    else:
                        pass

        if dims == check:  # there's only one document
            for i, dim in enumerate(dims):
                if type(dim[0]) in dtypes.numbers:
                    # invent some dimensions
                    # the prediction coming from these dimensions will not be used anyway, since it is x
                    if type(dim[0]) in dtypes.floats:
                        dim[0] = dim[0] - 0.05 * dim[0]
                        dim[1] = dim[1] + 0.05 * dim[1]
                    elif type(dim[0]) in dtypes.ints:
                        dim[0] = dim[0] - 1
                        dim[1] = dim[1] + 1

                    if dim[0] > dim[1]:
                        dim = [dim[1], dim[0]]

                    dims[i] = dim

        dims = [tuple(dim) for dim in dims]
        return dims

    def run_task(self, fw_spec):
        """
        The fireworks' FireTask implementation of running the optimization loop.

        :param fw_spec: (dict) the firetask spec. Must contain a '_y' key with a float type field and must contain
        a '_z' key containing a vector uniquely defining the search space.

        :return: (FWAction object) a fireworks-interpretable object for creating a new, updated workflow.
        """

        z = fw_spec['_z']
        y = fw_spec['_y']
        Z_dims = [tuple(dim) for dim in self['dimensions']]
        wf_creator = self._deserialize_function(self['wf_creator'])
        wf_creator_args = self['wf_creator_args'] if 'wf_creator_args' in self else {}

        if not isinstance(wf_creator_args, dict):
            raise TypeError("wf_creator_args should be a dictonary of keyword arguments.")

        # define the function which can fetch X
        get_x = self._deserialize_function(self['get_x']) if 'get_x' in self else lambda *args, **kwargs : []
        x = get_x(z)

        # _store the data
        id = self._store({'z':z, 'y':y, 'x':x}).inserted_id

        # gather all docs from the _collection in a concurrency-friendly manner
        Z_ext = []
        Y = []
        for doc in self._collection:
            if all (k in doc for k in ('x','y','z')):
                Z_ext.append(doc['z'] + doc['x'])
                Y.append(doc['y'])

        # extend the dimensions to X features, so that X information can be used in optimization
        Z_ext_dims = Z_dims + self._X_dims if x != [] else Z_dims

        # run machine learner on Z and X features
        predictor = 'forest_minimize' if not 'predictor' in self else self['predictor']

        if predictor in self.optimizers:
            import skopt
            z_total_new = getattr(skopt, predictor)(lambda x:0, Z_ext_dims, x0=Z_ext, y0=Y, n_calls=1,
                                                            n_random_starts=0).x_iters[-1]
        else:
            try:
                predictor_fun = self._deserialize_function(predictor)
                z_total_new = predictor_fun(Z_ext, Y, Z_ext_dims)

            except:
                raise ValueError("The custom predictor function {fun} did not call correctly! "
                                 "The arguments were: \n arg1: list of {arg1len} lists of {arg1} \n"
                                 "arg2: list {arg2} of length {arg2len} \n arg3: {arg3}"
                                 .format(fun=predictor, arg1=type(Z_ext[0][0]), arg1len=len(Z_ext), arg2=type(Y[0]),
                                         arg2len=len(Y), arg3=Z_ext_dims))

        # remove X features from the new Z vector
        z_new = z_total_new[0:len(z)]

        # duplicate checking. makes sure no repeat z vectors are inserted into the turboworks _collection
        if 'duplicate_check' in self:
            if self['duplicate_check']:
                if self._is_discrete(Z_dims):
                    z_new = self._dupe_check(z, Z_dims)

        self._store({'z_new':z_new, 'z_total_new':z_total_new}, update=True, id=id)

        # return a new workflow
        return FWAction(additions=wf_creator(z_new,**wf_creator_args))
