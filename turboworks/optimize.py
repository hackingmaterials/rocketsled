"""
The main FireTasks for running automatic optimization loops are contained in this module.
"""


from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from fireworks import FWAction
from pymongo import MongoClient
from turboworks.references import dtypes
from turboworks.discrete import calculate_discrete_space
import skopt

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

    :param predictor: (string) names a function which given a list of inputs, a list of outputs, and a dimensions space,
     can return a new optimized input vector. Can specify either a skopt function or a custom function.
        example: predictor = 'my_module.my_predictor'

    :param duplicate_check: (boolean) If True, checks for duplicate guesss in discrete, finite spaces. (NOT currently
    working with concurrent workflows). Default is no duplicate check.
    """

    _fw_name = "OptTask"
    required_params = ['wf_creator', 'dimensions']
    optional_params = ['get_x', 'predictor', 'duplicate_check']

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
        self._tw_meta_collection = self._tw_db.meta
        self.optimizers = ['gbrt_minimize', 'dummy_minimize', 'forest_minimize', 'gp_minimize']

    def store(self, spec, update = False, id = None):
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

    def deserialize_function(self, fun):
        """
        Takes a fireworks serialzed function handle and maps it to a function object.

        :param fun: (String) a 'module.function' style string specifying the function
        :return: (function) the function object defined by fun
        """
        toks = fun.rsplit(".", 1)
        if len(toks) == 2:
            modname, funcname = toks
            mod = __import__(modname, globals(), locals(), [str(funcname)], 0)
            return getattr(mod, funcname)

    def attr_exists(self, attr):
        """
        Test if a dict-style attribute of the FireTask object was defined in the constructor as a kwarg.

            for example:
                o = OptTask(wf_creator = 'mymod.myfun', dimensions = [(1,100)], get_x = 'mymod.get_x')
                o.attr_exists('predictor') --> returns False, since the predictor kwarg was not given.

        :param attr: (String) the name of an attrbute for the FireTask.
        :return: (boolean) whether the attribute as kwarg was given to the FireTask or not.
        """

        try:
            self[attr]
            return True
        except(KeyError):
            return False

    def is_discrete(self, dims):
        """
        Checks if the search space is totally discrete.

        :param dims: (list of tuples) dimensions of the search space
        :return: (boolean) whether the search space is totally discrete.
        """

        for dim in dims:
            if type(dim[0]) not in dtypes.discrete or type(dim[1]) not in dtypes.discrete:
                return False
        return True

    def populate_meta(self, dims):
        """
        Populates the turboworks meta database. Used for error checking and giving new guesses if optimizer repeats.

        :param dims: (list of tuples)
        :return: None
        """
        space = calculate_discrete_space(dims)
        for z in space:
            if self._tw_meta_collection.find({'z':z}).count() == 0: # if the guess is not already in the db
                self._tw_meta_collection.insert_one({'z':z, 'guessed':'no'})

    def dupe_check(self, z):
        '''
        Checks to see if a unique input vector z has already been run for the database.
        Currently only works for workflows acessing the database sequentially.

        :param z: list of categorical values or integers specifying a unique point in the input space.
            example: z = [1,2,3,'orange','dog']

        :return: z, the new z vector. if z was already tried, z is a random guess from the remaining untried space.
        '''
        # todo: duplicate checking does not work with multiple processes
        # todo: find a better, less storage/write/read intensive method that is also multiprocess compat.

        available_z = self._tw_meta_collection.find({'z':z, 'guessed':'no'})

        if available_z.count() > 0:
            # the z is available and is in the db, check all concurrently written instances and update them
            self._tw_meta_collection.update_many({'z':z, 'guessed':'no'}, {'$set':{'guessed':'yes'}})
        else:
            # the z is not available because it is not in the db, randomly sample remaining space for new z
            new_doc = self._tw_meta_collection.find_one({'guessed':'no'})
            z = new_doc['z']
            self._tw_meta_collection.update_many({'z':z, 'guessed':'no'}, {'$set':{'guessed':'yes'}})

        if self.meta_exhausted:  # the db has been exhausted of choices
            raise ValueError("The search space has been exhausted.")

        return z

    @property
    def meta_exhausted(self):
        """
        Checks if the turboworks meta database is exhausted (i.e., all possible guesses in the space have been guessed).

        :return: (boolean) whether all the guesses in the space have been guessed.
        """
        return True if self._tw_meta_collection.find({'guessed': 'no'}).count() == 0 else False

    @property
    def meta_empty(self):
        """
        Checks if the turboworks meta database is empty (i.e., it's been reset for a new workflow or optimization).

        :return: (boolean) whether the database is empty
        """
        return True if self._tw_meta_collection.count() == 0 else False

    @property
    def collection(self):
        """
        Wrapper of .find() pymongo method for easy access to most up to date collection.

        :return: (PyMongo cursor object) the results of an empty turboworks database query.
        """

        return self._tw_collection.find()

    @property
    def X_dims(self):
        """
        Creates some X dimensions so that the optimizer can run without the user specifing the X dimension range.
        Simply sets each dimension equal to the (lowest, highest) values of any x for that dimension in the database.

        If there is only one document in the database, it sets the dimension to slightly higher and lower values than
        the x dimension value.

        For categorical dimensions, it includes all dimensions in X.

        :return: (list of tuples) a list of dimensions
        """

        X = [doc['x'] for doc in self.collection]
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
        wf_creator = self.deserialize_function(self['wf_creator'])

        # define the function which can fetch X
        get_x = self.deserialize_function(self['get_x']) if self.attr_exists('get_x') else lambda *args, **kwargs : []
        x = get_x(z)

        # store the data
        id = self.store({'z':z, 'y':y, 'x':x}).inserted_id

        # gather all docs from the collection in a concurrency-friendly manner
        Z_ext = []
        Y = []
        for doc in self.collection:
            if all (k in doc for k in ('x','y','z')):
                Z_ext.append(doc['z'] + doc['x'])
                Y.append(doc['y'])

        # extend the dimensions to X features, so that X information can be used in optimization
        Z_ext_dims = Z_dims + self.X_dims if x != [] else Z_dims

        # run machine learner on Z and X features
        predictor = 'forest_minimize' if not self.attr_exists('predictor') else self['predictor']

        if predictor in self.optimizers:
                z_total_new = getattr(skopt, predictor)(lambda x:0, Z_ext_dims, x0=Z_ext, y0=Y, n_calls=1,
                                                                n_random_starts=0).x_iters[-1]
        else:
            try:
                predictor_fun = self.deserialize_function(predictor)
                z_total_new = predictor_fun(Z_ext, Y, Z_ext_dims)

            except:
                raise ValueError("The custom predictor function {fun} did not call correctly! "
                                 "The arguments were: \n arg1: list of {arg1len} lists of {arg1} \n"
                                 "arg2: list {arg2} of length {arg2len} \n arg3: {arg3}"
                                 .format(fun=predictor, arg1=type(Z_ext[0][0]), arg1len=len(Z_ext), arg2=type(Y[0]),
                                         arg2len=len(Y), arg3=Z_ext_dims))

        # remove X features from the new Z vector
        z_new = z_total_new[0:len(z)]

        # duplicate checking. makes sure no repeat z vectors are inserted into the turboworks collection
        if self.attr_exists('duplicate_check'):
            if self['duplicate_check']:
                if self.is_discrete(Z_dims):
                    if self.meta_empty:
                        self.populate_meta(Z_dims)
                        self._tw_meta_collection.update_many({'z':z}, {'$set':{'guessed':'yes'}})

                    z_new = self.dupe_check(z_new)
                    z_total_new = z_new + z_total_new[len(z_new):]

        self.store({'z_new':z_new, 'z_total_new':z_total_new}, update=True, id=id)

        # return a new workflow
        return FWAction(additions=wf_creator(z_new))

