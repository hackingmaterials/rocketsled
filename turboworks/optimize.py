"""
The FireTask for running automatic optimization loops are contained in this module.
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

    OptTask takes in and stores a vector 'z' which uniquely defines the input space and a scalar 'y' which is the
    scoring metric. OptTask produces a new 'z' vector to minimize 'y' using information from all 'z' vectors and 'y'
    scalars. Additionally, an 'x' vector of extra features can be used by OptTask to better optimize.

    Attributes:
        wf_creator (function): returns a workflow based on a unique vector, z.
        dimensions ([tuple]): each 2-tuple in the list defines the search space in (low, high) format.
            For categorical dimensions, includes all possible categories as a list.
            Example: dimensions = dim = [(1,100), (9.293, 18.2838), ("red", "blue", "green")].
        get_x (string): the fully-qualified name of a function which, given a z vector, returns another vector x which
            provides extra information to the machine learner. The features defined in x are not used to run the
            workflow creator.
            Examples: 
                get_x = 'my_module.my_fun'
                get_x = '/path/to/folder/containing/my_package.my_module.my_fun'
        predictor (string): names a function which given a list of inputs, a list of outputs, and a dimensions space,
            can return a new optimized input vector. Can specify either a skopt function or a custom function.
            Example: predictor = 'my_module.my_predictor'
        wf_creator_args (dict): details the kwargs to be passed to the wf_creator function alongside the z vector
        duplicate_check (bool): If True, checks for duplicate guesss in discrete, finite spaces. (NOT currently 100%
            working with concurrent workflows). Default is no duplicate check.
        host (string): The name of the MongoDB host where the optimization data will be stored. The default is
            'localhost'.
        port (int): The number of the MongoDB port where the optimization data will be stored. The default is 27017.
        name (string): The name of the MongoDB database where the optimization data will be stored.
        opt_label (string): Names the collection of that the particular optinization's data will be stored in. Multiple
            collections correspond to multiple independent optimization.
    """

    _fw_name = "OptTask"
    required_params = ['wf_creator', 'dimensions']
    optional_params = ['get_x', 'predictor', 'wf_creator_args', 'duplicate_check', 'host', 'port', 'name', 'opt_label']


    def _store(self, spec, update = False, id = None):
        """
        Stores and updates turboworks database files.
        
        Args:
            spec (dict): a turboworks-generated spec (or subset of a spec) to be stored in the turboworks db.
            update (bool): whether to update the document (True) or insert a new one (False)
            id (ObjectId): the PyMongo BSON id object. if update == True, updates the document with this id.
            
        Returns:
            (ObjectId) the PyMongo BSON id object for the document inserted/updated.
        """

        if update == False:
            return self.collection.insert_one(spec)
        else:
            return self.collection.update({"_id":id },{'$set' : spec})

    def _deserialize_function(self, fun):
        """
        Takes a fireworks serialzed function handle and maps it to a function object.

        Args:
            fun (string): a 'module.function' or '/path/to/mod.func' style string specifying the function
         
        Returns:
            (function) The function object defined by fun
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

        Args:
            dims ([tuple]): dimensions of the search space
            
        Returns:
            (bool) whether the search space is totally discrete.
        """

        for dim in dims:
            if type(dim[0]) not in dtypes.discrete or type(dim[1]) not in dtypes.discrete:
                return False
        return True

    def _calculate_discrete_space(self, dims):
        """
        Calculates all entries in a discrete space.

        Example:

            >>> dims = [(1,2), ["red","blue"]]
            >>> space = _calculate_discrete_space(dims)
            >>> space
            [(1, 'red'), (1, 'blue'), (2, 'red'), (2, 'blue')]

        Args:
            dims ([tuple]): dimensions of the search space.

        Returns:
            ([list]) all possible combinations inside the discrete space
        """

        total_dimspace = []

        for dim in dims:
            if type(dim[0]) in dtypes.ints:
                # Then the dimension is of the form (lower, upper)
                lower = dim[0]
                upper = dim[1]
                dimspace = list(range(lower, upper + 1))
            elif type(dim[0]) in dtypes.floats:
                raise ValueError("The dimension is a float. The dimension space is infinite.")
            else:  # The dimension is a discrete finite string list
                dimspace = dim
            total_dimspace.append(dimspace)

        return [[x] for x in total_dimspace[0]] if len(dims)==1 else list(itertools.product(*total_dimspace))

    def _dupe_check(self, z, Z_dim):
        """
        Check for duplicates so that expensive workflow will not be needlessly rerun.

        Args:
            z (list): input to be duplicate checked
            Z_dim ([tuples]): space in which to check for duplicate

        Returns:
            (list) updated input which is either the duplicate-checked input z or a randomly picked replacement
        """

        # todo: available_z should be stored per job, so it does not have to be created more than once.
        # TODO: I would agree that a performance improvement is needed, e.g. by only computing the full discrete space as well as available z only once (-AJ)
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
        the x dimension value. For categorical dimensions, it includes all dimensions in X.

        Returns:
            ([tuple]) a list of dimensions
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
        FireTask implementation of running the optimization loop.

        Args:
            fw_spec (dict): the firetask spec. Must contain a '_y' key with a float type field and must contain
                a '_z' key containing a vector uniquely defining the search space.

        Returns:
            (FWAction)
        """
        # TODO: I am confused about the notation; usually we should use y (output) and X (all inputs, usually capital b/c it is a vector) in machine learning. The z is a bit confusing. I would suggest that z->x or z->X (I actually suggest lowercase so people don't get confused about is lower and upper case). Then your original x becomes x_added or x_user or something. (-AJ)
        z = fw_spec['_z']  # TODO: in retrospect, we should probably have this be fw_spec["_tbw_z"]. That way, all the Turboworks parameters are clearly labeled and separated from anything else the user wants to do (-AJ).
        y = fw_spec['_y']  # TODO: in retrospect, we should probably have this be fw_spec["_tbw_y"]. That way, all the Turboworks parameters are clearly labeled and separated from anything else the user wants to do (-AJ).
        Z_dims = [tuple(dim) for dim in self['dimensions']]  # TODO: I don't understand the point of this (-AJ)
        wf_creator = self._deserialize_function(self['wf_creator'])

        wf_creator_args = self['wf_creator_args'] if 'wf_creator_args' in self else {}  # TODO: call it "wf_creator_kwargs" if these are supposed to be keyword args. You can have another one called "wf_creator_args" which would be an array of **args. (-AJ)
        if not isinstance(wf_creator_args, dict):
            raise TypeError("wf_creator_args should be a dictonary of keyword arguments.")

        opt_label = self['opt_label'] if 'opt_label' in self else 'opt_default'
        # TODO: if left empty, this should default to a string uniquely representing the fireworks workflow.
        # TODO: once integrated with Fireworks, default opt_label to _fw_name
        # TODO: If I understand correctly, *all* workflows within one optimization experiment (e.g., perhaps 1000 workflows) should have the same opt_label. I would in advise against any kind of cute thing like defaulting to fw_name in that case. Try to use principle of least surprise. (-AJ)

        host = self['host'] if 'host' in self else 'localhost'
        port = self['port'] if 'port' in self else 27017
        name = self['name'] if 'name' in self else 'turboworks'

        # TODO: for host, port, name, maybe talk to AJ. There should be two options: (i) the user sets these variables, in which case use those (already done in your solution). (ii) The user sends in a LaunchPad object to the Firework (ask AJ), in which case use the LaunchPad's fireworks db as the db. If neither of those, don't use localhost. Throw an error asking the user to specify the database using either of the two methods. (-AJ)

        mongo = MongoClient(host, port)
        db = getattr(mongo, name)
        self.collection = getattr(db, opt_label)

        # define the function which can fetch X
        # TODO: it would be less confusing if you simply didn't call get_x() if the parameter wasn't set instead of defining the lambda function. e.g., x = self._deserialize_function(self['get_x']) if 'get_x' in self else []  (-AJ)
        get_x = self._deserialize_function(self['get_x']) if 'get_x' in self else lambda *args, **kwargs : []
        x = get_x(z)

        # store the data
        id = self._store({'z':z, 'y':y, 'x':x}).inserted_id

        # gather all docs from the collection
        Z_ext = []
        Y = []
        # TODO: depending on how big the docs are in the collection apart from x,y,z, you might get better performance using find({}, {"x": 1, "y": 1, "z": 1})  (-AJ)
        # TODO: I would need to think whether the concurrency read is really done correctly (-AJ)
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
                z_total_new = predictor_fun(Z_ext, Y, Z_ext_dims)  #  TODO: later, you might want to add optional **args and **kwargs to this as well. For now I think it is fine as is. (-AJ)

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
