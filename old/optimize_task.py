"""
FireTask implementation of optimization algorithms.
"""

__author__ = "Alexander Dunn"
__email__ = "ardunn@lbl.gov"
__version__ = "0.1"

from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from pymongo import MongoClient
from collections import OrderedDict
from functools import reduce
import operator
from six import string_types
from skopt import gbrt_minimize
from turboworks.dummy import dummy_minimize
from fireworks import FWAction, Workflow



@explicit_serialize
class OptimizeTask(FireTaskBase):
    _fw_name = "OptimizeTask"

    required_params = []
    optional_params = []

    # COMPLETED METHODS
    # todo: write docstrings

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
        self.delimiter = '/'
        self.meta_fw_keys = ['_id', '_tasks']

        self.input_keys = []
        self.output_keys = []
        self.dimension_keys = []
        self.tw_spec = {}
        self.fw_spec_form = {}

    def run_task(self, fw_spec):
        # This method should be overridden
        raise NotImplementedError("You must have a run_task implemented!")

    #todo: fix _store so its automatic and keeps original spec form and add _auto_store!! -> tw_spec_to_fw_spec
    def store(self, spec):

        self.tw_spec = spec
        self._tw_collection.insert_one(OrderedDict(spec))

    def parse_compound_key(self, k):

        if type(k) is list:    #the parsing request has been sent from extract on behalf of update_input
            keys = k
        elif isinstance(k, basestring):
            if self.delimiter in k:
                keys = k.split(self.delimiter)  # a top down list of nested dictionary keys
            else:
                keys = [k]
        else:
            raise TypeError("parse_compound_key wasn't given a compound key or list")

        return keys

    def extract(self, k, d):
        '''

        :param k: a single key in class form, e.g. 'types.old.r'
        :param d: a nested dictionary to be searched
        :return: the desired value defined by k
        '''

        keys = self.parse_compound_key(k)
        return reduce(operator.getitem, keys, d)

    # Extracting compound keys from a single nested dictionary and storing in a new flat dictionary
    def flatten(self, k_list = None, d = None):

        if type(k_list) is not list and k_list is not None:
            if not isinstance(k_list[0], string_types):
                raise TypeError("Keys should be in list of strings form. For example, ['X', 'Y/b/z']")

        from pprint import pprint
        pprint(d)

        k_list = self.extend_compound_key_list(key_list=k_list, d=d)

        try:
            k_list = [x.encode('UTF8') for x in k_list]
        except:
            pass



        k_list = sorted(k_list, key=str.lower)

        flat = {}
        for k in k_list:
            flat[k] = self.extract(k,d)

        return flat

    # Extracting from a list of flat dictionaries (using compound keys)
    def flat_extract(self, k_list = None, d_list=None):
        if d_list is None:
            d_list = self._tw_collection.find()

        k_list = self.fix_keys(k_list=k_list)

        extract = []

        for doc in d_list:
            extract.append([doc[k] for k in k_list])

        # return self.flatten_nested_list(extract)
        return extract

    # Fix a list of compound keys
    def fix_keys(self, k_list = None, extension = False, dict_template = None):

        if extension:
            k_list = self.extend_compound_key_list(key_list=k_list, d=dict_template)

        try:
            k_list = [x.encode('UTF8') for x in k_list]
        except:
            pass

        return sorted(k_list, key=str.lower)

    # Extracting from a list of nested dictionaries
    # todo: not working
    def auto_extract(self, k_list = None, d_list=None, label="inputs", n=None):
        '''

        auto_extract processes documents in a mongo database to gather complete (or limited)
        sets of data for specified sets of input and output variables.

            Example:

            # define a nested dictionary
            my_dict = {
                'A':1200,
                'energy': 0.001,
                'types': {
                    'old':{
                        'q': 88,
                        'r': 99
                    },
                    'new':{
                        's': 65,
                        't': 53
                    }}}


            # insert 3 identical documents into a db
            for _ in range(2):
                db._collection.insertOne(my_dict)

            print(auto_extract(['A', 'types.old.q'], label='inputs'))

            [[1200, 12000, 1200], [88, 88, 88]]]

        :param k_list: (list) of possible (string) ML inputs which will be used as keys searching
        the turboworks database for all stored data. given in class/attr 'dot' form for nested
         dictionaries.

                Example: inputs = ['Structure.A', 'energy']

            If a nested dictionary is given as an input, it will gather all non-dict
            variables specified in that dictionary.

        :param label: (string) stores class attr under the label if label = "input" or label = "output"

        :param n: (int) number of documents to search

                Example: n = 2500

        :return: extract (dict) containing all datapoints from the database matching the inputs and
        outputs arguments. Each dict value is a list of lists; innermost lists are collections of
        a single feature across multiple documents.

        '''

        extract = list()

        if type(k_list) is not list and k_list is not None:
            if not isinstance(k_list[0], string_types):
                raise TypeError("Keys should be in list of strings form. For example, ['X', 'Y.b.z']")
        elif k_list is None:
            # key_scavenger in extend_compound_key already handles this
            pass

        if d_list is None:
            #todo: fix this?
            dict_template = self._tw_collection.find_one()
            d_list = self._tw_collection.find()

            if n is None:
                n = self._tw_collection.count()
        else:
            dict_template = d_list[0]

            if n is None:
                n = len(d_list)

        k_list = self.fix_keys(k_list = k_list, extension=True, dict_template=dict_template)

        if label in ['inputs', 'input', 'in', 'input_list', 'features']:
            self.input_keys = k_list
        elif label in ['outputs', 'output', 'out', 'output_list']:
            self.output_keys = k_list
        elif label in ['dimensions', 'dims', 'dim', 'input_dimensions', 'input_dims', 'input_dim']:
            self.dimension_keys = k_list

        for k in k_list:
            sublist = []

            #todo: check if _tw_collection.find() results in diff order when called
            for doc in d_list:
                if len(sublist) <= n-1:
                    sublist.append(self.extract(k,doc))
            extract.append(sublist)

        return self.flatten_nested_list(extract)

    # makes single feature/output extracts NOT as a nested list
    # todo: maybe not needed
    def flatten_nested_list(self, nested_list):

        # for example, if outputs = ['X'], extract is [32], NOT [[32]]
        if type(nested_list) is list:
            if len(nested_list) == 1:
                extract = nested_list[0]
                return extract

        return nested_list

    def update_input(self, updated_value, k, d=None):
        # updates a single input

        if d is None:
            d = self.tw_spec

        keys = self.parse_compound_key(k)

        if len(keys) == 1:
            self.tw_spec[keys[0]] = updated_value
        else:
            self.extract(keys[:-1],d)[keys[-1]] = updated_value

    def auto_update(self, updated_values, keys=None):
        # automatically associates a recently joined dictionary and puts it back in fw_spec form
        # takes an ordered list of class/attr dict strings, a list of new ML'd data for the guess
        # and it updates the fw_spec accordingly


        if keys is None:
            keys = sorted(self.input_keys, key = str.lower)
        if type(keys) is not list:
            raise TypeError("Keys should be in list form. For example, ['X', 'Y.b.z']")

        if type(updated_values) is list:
            for i, updated_value in enumerate(updated_values):

                if len(updated_values) != len(keys):
                    #if you're trying to write too few or too many data to the list of compound keys
                    raise TypeError("""
                                        The updated input does not have the same dimensions as the original input.
                                        Make sure to call auto_extract with label = 'output' so the outputs are not
                                        overwriting
                                        your original inputs.""")
                try:
                    self.update_input(updated_value, keys[i])

                except(KeyError):
                    raise ValueError("Keys should be the same as they were extracted with.")

    def key_scavenger(self, d, compound_key='', top_level=True, compound_keys=None, superkey = None):
        # returns all highest level non-dict entries in a list of class/attr dict style strings
        # used as a tool to return all values if a compound key ends in a dict.
        # for example selecting 'types.old' from ref_dict results in selecting 'types.old.q'
        # and 'types.old.r'

        if compound_keys is None:
            compound_keys = []

        for key in d:
            if type(d[key]) is dict:
                if top_level:
                    self.key_scavenger(d[key], compound_key=compound_key + key,
                                  top_level=False, compound_keys=compound_keys)
                else:
                    self.key_scavenger(d[key], compound_key=compound_key + self.delimiter + key,
                                  top_level=False, compound_keys=compound_keys)

            else:
                if top_level:
                    compound_keys.append(key)
                else:
                    compound_keys.append(compound_key + self.delimiter + key)

        if top_level:
            if superkey is None:
                return compound_keys
            else:
                return [superkey + self.delimiter + key for key in compound_keys]

    def extend_compound_key_list(self, key_list = None, d = None):
        #expands a list of compound keys to include subkeys if not explicitly mentioned

        if key_list is None:
            # should get the entire dictionary...
            key_list = self.key_scavenger(d)

            # ...except if its a meta_fw_key
            key_list = [key for key in key_list if key not in self.meta_fw_keys]

        if d is None:
            # to preserve argument order
            raise TypeError("extend_compound_key requires a dictionary as input.")

        for key in key_list:
            sub = self.extract(key, d)
            if type(sub) is dict:
                key_list.remove(key)
                key_list.extend(self.key_scavenger(sub, superkey=key))
            else: pass

        return key_list

    def skopt_dummy(self, inputs):
        """
        Allows for the current skopt API to be used for sequential optimization without skopt modification
        :param inputs: skopt requires this
        :return: any number, does not matter because skopt is not storing this value and its not used elsewhere
        """
        return 0

    def tw_spec_to_fw_spec(self):
        pass

@explicit_serialize
class AutoOptimizeTask(OptimizeTask):
    required_params = ['workflow', 'inputs', 'outputs', 'dimensions']
    optional_params = ['initialized']

    _fw_name = "AutoOptimizeTask"

    def create_dict_by_fw(self, wf_dict):
        dict_by_fw = {}
        for fw in wf_dict["fws"]:
            name = fw["name"]
            spec = fw["spec"]

            if name in dict_by_fw:
                last_clone = self.get_last_clone_number(name, dict_by_fw)
                name = name + str(last_clone + 1)

            dict_by_fw[name] = spec

        return dict_by_fw

    #todo: this doesn't work if the d has a number in front of it...
    def get_last_clone_number(self, k, d):
        biggest = 0
        for i in d:
            if k in i:
                try:
                    num = int(i.replace(k, "", 1))
                    if num > biggest:
                        biggest = num
                except:
                    pass

        return biggest

    @property
    def is_initialized(self):
        try:
            self["initialized"]
            return True
        except:
            return False

    def run_task(self, fw_spec):

        wf_dict = self['workflow']
        wf = Workflow.from_dict(wf_dict)

        input_keys = self['inputs']
        output_keys = self['outputs']
        dim_keys = self['dimensions']

        all_keys = input_keys + output_keys + dim_keys


        flat = {}
        # grab all relevant values from the specs of each fw using the input, output, and dim keys
        from pprint import pprint
        # pprint(wf_dict["fws"][0])

        if self.is_initialized:

            dict_by_fw = self.create_dict_by_fw(wf_dict)
            # pprint(dict_by_fw)

            # ['Firework1.Structure.A', 'Firework2.Structure.B']


            from pprint import pprint
            pprint(wf_dict)

            tw_spec = self.flatten(k_list=all_keys, d=dict_by_fw)
            # {'Firework1.Structure.A': 1.31, 'Firework2.Structure.B': 2.03, etc.}
            # _store all the values in a document in Mongo in flat dict with upper.lower.etc keys
            self.store(tw_spec)

            # make the pretty X,y matrices
            X = self.flat_extract(k_list=input_keys)
            y = self.flat_extract(k_list=output_keys)
            print X
            print y


            # run random forest
            # associate the new x with the compound keys
            # update all the specs of the workflow with their nested values


        # return a FWAction(additons=wf) where wf includes a new AutoOptimizeTask with initialized = true
        wf_dict["fws"][0]["spec"]["_tasks"].append(AutoOptimizeTask(initialized = True, inputs = input_keys,
                                                                    outputs = output_keys, dimensions=dim_keys,
                                                                    workflow=wf))
        wf = Workflow.from_dict(wf_dict)

        return FWAction(additions=wf)




@explicit_serialize
class OptimizeTaskFromVector(FireTaskBase):

    required_params = ['input', 'output', 'dimensions']
    optional_params = []

    _fw_name = "OptimizeTaskFromVector"

    def __init__(self):
        self._tw_port = 27017
        self._tw_host = 'localhost'
        self._tw_mongo = MongoClient(self._tw_host, self._tw_port)
        self._tw_db = self._tw_mongo.turboworks
        self._tw_collection = self._tw_db.turboworks2

    def run_task(self, fw_spec):

        x = self['input']
        yi = self['output']
        d = self['dimensions']

        self.store(x, yi, d)
        X = self.get_field('input')
        y = self.get_field('output')

        print "got here"

        x_new = gbrt_minimize(self.dummy, d, x0=X, y0=y, n_calls=1, n_random_starts=0)

        print x_new

        return FWAction(update_spec={'input':x_new})


    def store(self, input, output, dimensions):
        store_dict = {'input':input, 'output':output, 'dimensions':dimensions}
        self._tw_collection.insert_one(store_dict)


    def get_field(self, field):

        biglist = []
        for doc in self._tw_collection.find():
            biglist.append(doc[field])

        return biglist

    def dummy(self, inputs):
        return 0



# my_wf = [fw1, fw2, fw3]
# should add AutoOptimizeTask onto fw3

class Visualize(object):
    def __init__(self):
        pass



