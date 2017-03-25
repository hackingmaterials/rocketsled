"""
The main FireTasks for running automatic optimization loops are contained in this module.
"""

import sys
import itertools
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from fireworks import FWAction
from pymongo import MongoClient
from references import dtypes

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

    :param opt_label: (string) Describes the optimization being run. If multiple optimizations have been run, this
    differentiates the runs so the data is not mixed together. The defaul is 'Unnamed'
    """

    _fw_name = "OptTask"
    required_params = ['wf_creator', 'dimensions']
    optional_params = ['get_x', 'predictor', 'wf_creator_args', 'duplicate_check', 'host', 'port', 'name', 'opt_label']


    def _store(self, spec, update = False, id = None):
        """
        Stores and updates turboworks database files.

        :param spec: (dict) a turboworks-generated spec (or sub-spec) to be stored in the turboworks db.
        :param update: (boolean) whether to update the document (True) or insert a new one (False)
        :param id: (ObjectId BSON object) the mongodb id object. if update == True, updates the document with this id.
        :return: (ObjectId BSON object) the mongodb id object for the document inserted/updated.
        """

        if update == False:
            return self.collection.insert_one(spec)
        else:
            return self.collection.update({"_id":id },{'$set' : spec})

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

    def _calculate_discrete_space(self, dims):
        """
        Calculates all entries in a discrete space.
        For example, if the dimensions are

         [(1,2), ["red","blue"]]

        _calculate_discrete_space will return all possible combinations of these dimensions' entries:

        [(1, 'red'), (1, 'blue'), (2, 'red'), (2, 'blue')]

        In duplicate checking for discrete spaces, the generated list will be narrowed down until no entries remain.

        WARNING: Very large discrete spaces will cause a memory bomb. Typically a space of about 1,000 entries takes
        0.005s to compute, but larger spaces can take much longer (or may just hog your RAM, be careful).

        :param dims: (list of tuples) dimensions of the search space.
        :return: (list of lists) all possible combinations inside the discrete space

        """
        total_dimspace = []

        for dim in dims:
            if type(dim[0]) in dtypes.ints:
                # Then the dimension is of the form (lower, upper)
                lower = dim[0]
                upper = dim[1]
                dimspace = list(range(lower, upper + 1))
            elif type(dim[0]) in dtypes.floats:
                # The chance of a random sample of identical float is nil
                raise ValueError("The dimension is a float. The dimension space is infinite.")
            else:  # The dimension is a discrete finite string list
                dimspace = dim
            total_dimspace.append(dimspace)

        return [[x] for x in total_dimspace[0]] if len(dims)==1 else list(itertools.product(*total_dimspace))

    def _dupe_check(self, z, Z_dim):
        """
        Check for duplicates so that expensive workflow will not be needlessly rerun.

        :param z: (list) input to be duplicate checked
        :param Z_dim: (list of tuples) space in which to check for duplicate
        :return: (list) updated input which is either the duplicate-checked input z or a randomly picked replacement
        """

        # todo: available_z should be stored per job, so it does not have to be created more than once.

        available_z = self._calculate_discrete_space(Z_dim)   # all possible choices in the discrete space

        for doc in self.collection.find():
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
    def _X_dims(self):
        """
        Creates some X dimensions so that the optimizer can run without the user specifing the X dimension range.
        Simply sets each dimension equal to the (lowest, highest) values of any x for that dimension in the database.

        If there is only one document in the database, it sets the dimension to slightly higher and lower values than
        the x dimension value.

        For categorical dimensions, it includes all dimensions in X.

        :return: (list of tuples) a list of dimensions
        """

        X = [doc['x'] for doc in self.collection.find()]
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

        opt_label = self['opt_label'] if 'opt_label' in self else 'opt_default'
        # TODO: if left empty, this should default to a string uniquely representing the fireworks workflow.
        # TODO: once integrated with Fireworks, default opt_label to _fw_name

        host = self['host'] if 'host' in self else 'localhost'
        port = self['port'] if 'port' in self else 27017
        name = self['name'] if 'name' in self else 'turboworks'

        mongo = MongoClient(host, port)
        db = getattr(mongo, name)
        self.collection = getattr(db, opt_label)

        # define the function which can fetch X
        get_x = self._deserialize_function(self['get_x']) if 'get_x' in self else lambda *args, **kwargs : []
        x = get_x(z)

        # _store the data
        id = self._store({'z':z, 'y':y, 'x':x}).inserted_id

        # gather all docs from the collection
        Z_ext = []
        Y = []
        for doc in self.collection.find():
            if all (k in doc for k in ('x','y','z')):  # concurrency read protection
                Z_ext.append(doc['z'] + doc['x'])
                Y.append(doc['y'])

        # extend the dimensions to X features, so that X information can be used in optimization
        Z_ext_dims = Z_dims + self._X_dims if x != [] else Z_dims

        # run machine learner on Z and X features
        predictor = 'forest_minimize' if not 'predictor' in self else self['predictor']

        if predictor in ['gbrt_minimize', 'random_guess', 'forest_minimize', 'gp_minimize']:
            import skopt
            z_total_new = getattr(skopt, predictor)(lambda x:0, Z_ext_dims, x0=Z_ext, y0=Y, n_calls=1,
                                                            n_random_starts=0).x_iters[-1]
        else:
            try:
                predictor_fun = self._deserialize_function(predictor)
                z_total_new = predictor_fun(Z_ext, Y, Z_ext_dims)

            except Exception as E:
                raise ValueError("The custom predictor function {} did not call correctly! \n {}".format(predictor,E))

        # separate 'predicted' X features from the new Z vector
        z_new, x_new = z_total_new[:len(z)], z_total_new[len(z):]

        # duplicate checking. makes sure no repeat z vectors are inserted into the turboworks collection
        if 'duplicate_check' in self:
            if self['duplicate_check']:
                if self._is_discrete(Z_dims):
                    z_new = self._dupe_check(z, Z_dims)
                    # do not worry about mismatch with x_new, as x_new is not used for any calculations

        self._store({'z_new':z_new, 'x_new':x_new}, update=True, id=id)

        # return a new workflow
        return FWAction(additions=wf_creator(z_new,**wf_creator_args))
