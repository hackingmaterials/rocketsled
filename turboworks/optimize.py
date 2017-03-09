from __future__ import print_function

from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from pymongo import MongoClient
from turboworks.references import dtypes
from turboworks.discrete import calculate_discrete_space
from fireworks import FWAction
import warnings
import skopt


@explicit_serialize
class OptTask(FireTaskBase):

    _fw_name = "OptTask"
    required_params = ['wf_creator', 'dimensions']
    optional_params = ['get_x', 'predictor', 'n_guesses']


    def __init__(self, *args, **kwargs):
        super(FireTaskBase, self).__init__(*args, **kwargs)

        #todo: make this work with fw style dictionaries?
        #todo: cleanup attrs + make this not so horrible
        #todo: add constructor arg for a new workflow

        self._tw_port = 27017
        self._tw_host = 'localhost'
        self._tw_mongo = MongoClient(self._tw_host, self._tw_port)
        self._tw_db = self._tw_mongo.turboworks
        self._tw_collection = self._tw_db.turboworks
        self._tw_meta_collection = self._tw_db.meta
        self._tw_space_exhausted = False

        self.optimizers = ['gbrt_minimize', 'dummy_minimize', 'forest_minimize', 'gp_minimize']

    def store(self, spec, update = False, id = None):

        if update == False:
            return self._tw_collection.insert_one(spec)
        else:
            return self._tw_collection.update({"_id":id },{'$set' : spec})

    def deserialize_function(self, fun):
        toks = fun.rsplit(".", 1)
        if len(toks) == 2:
            modname, funcname = toks
            mod = __import__(modname, globals(), locals(), [str(funcname)], 0)
            return getattr(mod, funcname)

    def attr_exists(self, attr):
        try:
            self[attr]
            return True
        except(KeyError):
            return False

    def is_discrete(self, dims):
        for dim in dims:
            if type(dim[0]) not in dtypes.discrete or type(dim[1]) not in dtypes.discrete:
                return False
        return True

    def populate_meta(self, dims):
        space = calculate_discrete_space(dims)
        for z in space:
            self._tw_meta_collection.insert_one({'vector':z})

    def dupe_check(self, z):
        available_z = self._tw_meta_collection.find({'vector':z})
        if z in available_z:
            # the z is available and is in the db
            self._tw_meta_collection.delete_one({'vector':z})
        else:
            # the z is not available because it is not in the db
            new_doc = self._tw_meta_collection.find_one()
            z, _id = new_doc['vector'], new_doc["_id"]
            self._tw_meta_collection.delete_one({"_id":_id})

        if self._tw_meta_collection.count() == 0:  # the db has been exhausted of choices
            warnings.warn("The dimensions space has been exhausted of points."
                          "The workflow will be restarted with using the last guess.")
            self._tw_space_exhausted = True
            print("z in dupe check".format(z))
            self._tw_meta_collection.insert_one({'vector': z})

        return z


    @property
    def collection(self):
        return self._tw_collection.find()

    @property
    def meta_empty(self):
        return True if self._tw_meta_collection.count() == 0 else False

    @property
    def X_dims(self):
        X = [doc['x'] for doc in self.collection]
        dims = [[x, x] for x in X[0]]
        check = dims

        for x in X:
            for i, dim in enumerate(dims):
                if x[i] < dim[0]:
                    # this value is the new minimum
                    dims[i][0] = x[i]
                elif x[i] > dim[1]:
                    # this value is the new maximum
                    dims[i][1] = x[i]
                else:
                    pass

        if dims == check: #there's only one document
            for i, dim in enumerate(dims):
                dim[0] = dim[0] - 0.05 * dim[0]
                dim[1] = dim[1] + 0.05 * dim[1]

                if dim[0] > dim[1]:
                    dim = [dim[1], dim[0]]

                dims[i] = dim

        dims = [tuple(dim) for dim in dims]
        return dims

    def run_task(self, fw_spec):

        z = fw_spec['_z']
        y = fw_spec['_y']
        Z_dims = [tuple(dim) for dim in self['dimensions']]
        wf_creator = self.deserialize_function(self['wf_creator'])

        # define the function which can fetch X
        get_x = self.deserialize_function(self['get_x']) if self.attr_exists('get_x') else lambda *args, **kwargs : []
        x = get_x(z)

        # store the data
        id = self.store({'z':z, 'y':y, 'x':x}).inserted_id

        # gather all docs from the collection
        Z_ext = [doc['z'] + doc['x'] for doc in self.collection]
        Y = [doc['y'] for doc in self.collection]

        # extend the dimensions to X features, so that skopt features will run
        Z_ext_dims = Z_dims + self.X_dims if x != [] else Z_dims

        # run machine learner on Z and X features
        if self.attr_exists('predictor'):

            if self['predictor'] in self.optimizers:
                z_total_new = getattr(skopt, self['predictor'])(lambda x:0, Z_ext_dims, x0=Z_ext, y0=Y, n_calls=1,
                                                                n_random_starts=0).x_iters[-1]
            else:
                try:
                    predictor = self.deserialize_function(self['predictor'])
                    z_total_new = predictor(Z_ext, Y, Z_ext_dims)

                except:
                    raise ValueError("The custom predictor function {fun} did not call correctly!"
                                    "The arguments were: \n arg1: list of {arg1len} lists of {arg1}"
                                    "arg2: list {arg2} of length {arg2len}"
                                    "arg3: {arg3}".format(fun=self['predictor'],
                                                         arg1=type(Z_ext[0][0]),
                                                         arg1len=len(Z_ext),
                                                         arg2=type(Y[0]),
                                                         arg2len=len(Y),
                                                         arg3=Z_ext_dims))

        else:
            # print("x", Z_ext)
            # print("y", Y)
            z_total_new = skopt.forest_minimize(lambda x:0, Z_ext_dims, x0=Z_ext, y0=Y, n_calls=1,
                                                n_random_starts=0).x_iters[-1]

        # remove X features from the new Z vector
        z_new = z_total_new[0:len(z)]


        # duplicate checking. makes sure no repeat z vectors are inserted into the turboworks collection
        if self.is_discrete(Z_dims):
            if self._tw_meta_collection.count() == 0:
                self.populate_meta(Z_dims)
                self._tw_meta_collection.delete_one({'vector':z})

            z_new = self.dupe_check(z_new)
            z_total_new = z_new + z_total_new[len(z_new):]

        if not self._tw_space_exhausted:
            self.store({'z_new':z_new, 'z_total_new':z_total_new}, update=True, id=id)
        else:
            # the space is exhausted and the guess is a repeat, and should not be stored.
            pass

        # return a new workflow
        return FWAction(additions=wf_creator(z_new))

