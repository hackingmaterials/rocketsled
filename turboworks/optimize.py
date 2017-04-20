"""
The FireTask for running automatic optimization loops are contained in this module.
"""

import sys
from itertools import product
from os import getpid
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from fireworks import FWAction
from pymongo import MongoClient, ReturnDocument
from references import dtypes
from time import sleep

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"


@explicit_serialize
class OptTask(FireTaskBase):
    """
    A FireTask for running optimization loops automatically.

    OptTask takes in and stores a vector 'x' which uniquely defines the input space and a scalar 'yi' which is the
    scoring metric. OptTask produces a new x vector to minimize y using information from all x vectors (X) and y
    scalars (Y). Additionally, a z vector of extra features can be used by OptTask to better optimize, although new
    values of z will be discarded.

    Required args:
        wf_creator (function): returns a workflow based on a unique vector, x.
        dimensions ([tuple]): each 2-tuple in the list defines the search space in (low, high) format.
            For categorical dimensions, includes all possible categories as a list.
            Example: dimensions = dim = [(1,100), (9.293, 18.2838), ("red", "blue", "green")].
    Optional args:
        get_z (string): the fully-qualified name of a function which, given a x vector, returns another vector z which
            provides extra information to the machine learner. The features defined in z are not used to run the
            workflow creator.
            Examples: 
                get_z = 'my_module.my_fun'
                get_z = '/path/to/folder/containing/my_package.my_module.my_fun'
        predictor (string): names a function which given a list of inputs, a list of outputs, and a dimensions space,
            can return a new optimized input vector. Can specify either a skopt function or a custom function.
            Example: predictor = 'my_module.my_predictor'
        max (bool): Makes optimization tend toward maximum values instead of minimum ones.
        wf_creator_args (list): details the positional args to be passed to the wf_creator function alongsize the z
            vector
        wf_creator_kwargs (dict): details the kwargs to be passed to the wf_creator function alongside the z vector
        duplicate_check (bool): If True, checks for duplicate guesss in discrete, finite spaces. (NOT currently 100%
            working with concurrent workflows). Default is no duplicate check.
        host (string): The name of the MongoDB host where the optimization data will be stored. The default is
            'localhost'.
        port (int): The number of the MongoDB port where the optimization data will be stored. The default is 27017.
        name (string): The name of the MongoDB database where the optimization data will be stored.
        lpad (LaunchPad): A Fireworks LaunchPad object.
        opt_label (string): Names the collection of that the particular optinization's data will be stored in. Multiple
            collections correspond to multiple independent optimization.
    """
    _fw_name = "OptTask"
    required_params = ['wf_creator', 'dimensions']
    optional_params = ['get_z', 'predictor', 'max', 'wf_creator_args', 'wf_creator_kwargs', 'duplicate_check',
                       'host', 'port', 'name', 'opt_label', 'lpad']

    def run_task(self, fw_spec):
        """
        FireTask for running an optimization loop.

        Args:
            fw_spec (dict): the firetask spec. Must contain a '_y_opt' key with a float type field and must contain
                a '_x_opt' key containing a vector uniquely defining the search space.

        Returns:
            (FWAction)
        """
        sleeptime = .001
        max_runs = 100000
        self._setup_db(fw_spec)
        x = fw_spec['_x_opt']
        yi = fw_spec['_y_opt']

        pid = getpid()

        for run in range(2*max_runs):
            manager_type = {'hold': {'$exists':1}, 'queue': {'$exists':1}}
            manager_docs = self.collection.find(manager_type)

            if manager_docs.count() == 0:
                self.collection.insert_one({'hold':pid, 'queue':[]})
            elif manager_docs.count() == 1:
                manager = self.collection.find_one(manager_type)
                manager_id = manager['_id']
                hold = manager['hold']

                if hold is None:
                    self.collection.find_one_and_update({'_id':manager_id}, {'$set':{'hold':pid}})

                elif hold != pid:
                    if pid not in manager['queue']:
                        new_queue = self.collection.find_one({'_id':manager_id})['queue']
                        new_queue.append(pid)
                        self.collection.find_one_and_update({'_id':manager_id}, {'$set':{'queue':new_queue}})
                    else:
                        sleep(sleeptime)

                elif hold == pid:

                    # type safety for dimensions to avoid cryptic skopt errors
                    x_dims = [tuple(dim) for dim in self['dimensions']]

                    # fetch additional attributes for constructing machine learning model by calling get_z, if it exists
                    z = self._deserialize_function(self['get_z'])(x) if 'get_z' in self else []

                    opt_id = self._store({'z': z, 'yi': yi, 'x': x})

                    # gather all docs from the collection
                    X_tot = []  # the matrix to store all x and z columns together
                    y = []  # TODO: prefer lowercase name y since this is a vector. See note below about a list comprehension to avoid problems.
                    for doc in self.collection.find({'x':{'$exists':1}, 'yi':{'$exists':1},'z':{'$exists':1}},
                                                    projection={'x': 1, 'yi': 1, 'z': 1}):
                        if all(k in doc for k in ('x', 'yi', 'z')):  # basic concurrency read 'protection'
                            # TODO: explain why the above is concurrency protection? Will we be missing 'z' if not? If so, then perhaps just test for z. This makes the code clearer. If not I need explanation...
                            X_tot.append(doc['x'] + doc['z'])
                            y.append(doc['yi'])

                    # change y vector if maximum is desired instead of minimum
                    max_on = self['max'] if 'max' in self else False
                    y = [-1 * yi if max_on else yi for yi in y]

                    # extend the dimensions to X features, so that X information can be used in optimization
                    X_tot_dims = x_dims + self._z_dims if z != [] else x_dims
                    # TODO: !!I really don't understand why _z_dims is needed. You are not optimizing over z, only x! Many combinations of z and x are anyway forbidden and should *not* be tested.  It might make a particular x look possibly good when it is not. This is an important point, please discuss w/me!!

                    # run machine learner on Z and X features
                    predictor = 'forest_minimize' if not 'predictor' in self else self['predictor']
                    if predictor in ['gbrt_minimize', 'random_guess', 'forest_minimize', 'gp_minimize']:
                        import skopt
                        predictor_fun = getattr(skopt, predictor)
                        predictor_data = predictor_fun(lambda x: 0, X_tot_dims, x0=X_tot, y0=y, n_calls=1,
                                                       n_random_starts=0)
                        x_tot_new = predictor_data.x_iters[-1]
                    else:
                        try:
                            predictor_fun = self._deserialize_function(predictor)
                            x_tot_new = predictor_fun(X_tot, y,
                                                      X_tot_dims)  # TODO: later, you might want to add optional **args
                            # and **kwargs to this as well. For now I think it is fine as is. (-AJ)

                        except Exception as E:
                            raise ValueError(
                                "The custom predictor function {} did not call correctly! \n {}".format(predictor, E))

                    # separate 'predicted' z features from the new x vector
                    x_new, z_new = x_tot_new[:len(x)], x_tot_new[len(
                        x):]  # TODO: this is subject to revision based on my 'important' comment about optimizing over z

                    # makes sure no repeat x vectors are inserted into the turboworks collection
                    if 'duplicate_check' in self:
                        if self['duplicate_check']:
                            if self._is_discrete(x_dims):
                                x_new = self._dupe_check(x, x_dims)

                    self._store({'z_new': z_new, 'x_new': x_new}, update=True, id=opt_id)

                    # udpdate the queue so that the oldest waiting process becomes active and is removed from the queue
                    queue =  self.collection.find_one({'_id':manager_id})['queue']

                    if queue == []:
                        self.collection.find_one_and_update({'_id': manager_id}, {'$set': {'hold': None}})
                    else:
                        new_hold, new_queue = queue[0], queue[1:]
                        self.collection.find_one_and_update({'_id': manager_id},
                                                            {'$set': {'hold': new_hold, 'queue': new_queue}})

                    # now that the queue is updated and the new guess has been calculated, start a new workflow
                    wf_creator = self._deserialize_function(self['wf_creator'])

                    wf_creator_args = self['wf_creator_args'] if 'wf_creator_args' in self else []
                    if not isinstance(wf_creator_args, list) or isinstance(wf_creator_args, tuple):
                        raise TypeError("wf_creator_args should be a list/tuple of positional arguments")

                    wf_creator_kwargs = self['wf_creator_kwargs'] if 'wf_creator_kwargs' in self else {}
                    if not isinstance(wf_creator_kwargs, dict):
                        raise TypeError("wf_creator_kwargs should be a dictonary of keyword arguments.")

                    return FWAction(additions=wf_creator(x_new, *wf_creator_args, **wf_creator_kwargs),
                                    update_spec={'optimization_id': opt_id})

            else:
                # there is more than one manager document, so reset it
                # todo: this can cause a loop where manager docs are being inserted/removed forever...should be improved
                self.collection.delete_many(manager_type)

            if run == max_runs:
                # an old process may be stuck on hold, so reset the manager and the queue/hold will repopulate
                # todo: this can cause concurrency problems if p1 is running ML and p2 times out, then 2+ processes
                # todo: (cont.) can be accessing the db at one time
                self.collection.find_one_and_update(manager_type, {'$set':{'hold': None,'queue':[]}})

        raise Exception("The manager is still stuck after resetting. Make sure no stalled processes are"
                        " in the queue.")





    def _store(self, spec, update=False, id=None):
        """
        Stores and updates turboworks database files.

        Args:
            spec (dict): a turboworks-generated spec (or subset of a spec) to be stored in the turboworks db.
            update (bool): whether to update the document (True) or insert a new one (False)
            id (ObjectId): the PyMongo BSON id object. if update == True, updates the document with this id.

        Returns:
            (ObjectId) the PyMongo BSON id object for the document inserted/updated.
        """


        if update:

            new_doc = self.collection.find_one_and_update({"_id": id}, {'$set': spec})
            return new_doc['_id']

        else:
            if 'duplicate_check' in self:
                if self['duplicate_check']:
                    # prevents errors when initial guesses are already in the database
                    x = spec['x']
                    new_doc =  self.collection.find_one_and_replace({'x':x}, spec, upsert=True,
                                                                    return_document= ReturnDocument.AFTER)
                    return new_doc['_id']
            else:
                return self.collection.insert_one(spec).inserted_id

    def _setup_db(self, fw_spec):
        '''
        Sets up a MongoDB database for storing optimization data.

        Args:
            fw_spec (dict): The spec of the Firework which contains this Firetask.

        Returns:
            None
        '''

        opt_label = self['opt_label'] if 'opt_label' in self else 'opt_default'
        db_reqs = ('host', 'port', 'name')

        # determine where Mondodb information will be stored
        if any(req in self for req in db_reqs):
            if all(req in self for req in db_reqs):
                host, port, name = [self[k] for k in db_reqs]
            else:
                raise AttributeError("Host, port, and name must all be specified!")

        elif 'lpad' in self:
            lpad = self['lpad']
            host, port, name = [lpad[req] for req in db_reqs]

        # todo: currently not working with multiprocessing objects!
        elif '_add_launchpad_and_fw_id' in fw_spec:
            if fw_spec['_add_launchpad_and_fw_id']:
                host, port, name = [getattr(self.launchpad, req) for req in db_reqs]

        # todo: add my_launchpad.yaml option via Launchpad.auto_load()?
        else:
            raise AttributeError("The optimization database must be specified explicitly (with host, port, and name)"
                                 " with a Launchpad object (lpad), or by setting _add_launchpad_and_fw_id to True on"
                                 " the fw_spec.")

        mongo = MongoClient(host, port)
        db = getattr(mongo, name)
        self.collection = getattr(db, opt_label)

    def _deserialize_function(self, fun):
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

        return [[x] for x in total_dimspace[0]] if len(dims) == 1 else list(product(*total_dimspace))

    def _dupe_check(self, x, x_dim):
        """
        Check for duplicates so that expensive workflow will not be needlessly rerun.

        Args:
            x (list): input to be duplicate checked
            x_dim ([tuples]): space in which to check for duplicate

        Returns:
            (list) updated input which is either the duplicate-checked input z or a randomly picked replacement
        """
        n_random_tries = 5

        if self.collection.find({'x': x}).count() == 0:
            # x is not in the collection
            return x
        else:
            # x is already in the collection
            import random

            random_tries = 0
            while True:
                randx = []
                for dim in x_dim:
                    dim_type = type(dim[0])
                    if dim_type in dtypes.discrete:
                        if dim_type in dtypes.ints:
                            randx.append(random.randint(dim[0], dim[1]))
                        elif dim_type in dtypes.others:
                            randx.append(random.choice(dim))
                    else:
                        raise TypeError("The dimension {} is not discrete. "
                                        "The guess cannot be duplicate checked.".format(dim))
                random_tries += 1

                if randx != x and self.collection.find({'x': randx}) == 0:
                    # randx is not in the collection, use it
                    return randx
                if random_tries == n_random_tries:
                    break

            # n_random_tries have been tried and its time to do an expensive duplicate check
            total_x = self._calculate_discrete_space(
                x_dim)  # all possible choices in the discrete space (expensive)

            for doc in self.collection.find({'x':{'$exists':1}, 'yi':{'$exists':1},'z':{'$exists':1}}):
                if tuple(doc['x']) in total_x:
                    total_x.remove(tuple(doc['x']))

            if len(total_x) == 0:
                raise ValueError("The search space has been exhausted.")

            if x in total_x:
                return x
            else:
                return random.choice(total_x)

    @property
    def _z_dims(self):
        """
        Creates some z dimensions so that the optimizer can run without the user specifing the z dimension range.
        Simply sets each dimension equal to the (lowest, highest) values of any z for that dimension in the database.
        If there is only one document in the database, it sets the dimension to slightly higher and lower values than
        the z dimension value. For categorical dimensions, it includes all dimensions in z.

        Returns:
            ([tuple]) a list of dimensions
        """

        Z = [doc['z'] for doc in self.collection.find({'x':{'$exists':1}, 'yi':{'$exists':1},'z':{'$exists':1}})]
        dims = [[z, z] for z in Z[0]]
        check = dims

        cat_values = []

        for z in Z:
            for i, dim in enumerate(dims):
                if type(z[i]) in dtypes.others:
                    # the dimension is categorical
                    if z[i] not in cat_values:
                        cat_values.append(z[i])
                        dims[i] = cat_values
                else:
                    if z[i] < dim[0]:
                        # this value is the new minimum
                        dims[i][0] = z[i]
                    elif z[i] > dim[1]:
                        # this value is the new maximum
                        dims[i][1] = z[i]
                    else:
                        pass

        if dims == check:  # there's only one document
            for i, dim in enumerate(dims):
                if type(dim[0]) in dtypes.numbers:
                    # invent some dimensions
                    # the prediction coming from these dimensions will not be used anyway, since it is z
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

