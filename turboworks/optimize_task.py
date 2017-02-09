"""
FireTask implementation of optimization algorithms.
"""

__author__ = "Alexander Dunn"
__email__ = "ardunn@lbl.gov"
__version__ = "0.1"

from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase, FWAction
from pymongo import MongoClient
from turboworks.gp_opt import gp_minimize
from turboworks.dummy_opt import dummy_minimize
import numpy as np
from collections import OrderedDict
import combo
from turboworks.discrete_spacify import calculate_discrete_space, duplicate_check
from contextlib import contextmanager
import sys, os
from functools import reduce
import operator
from six import string_types


#todo: remove this:
from pprint import pprint
from turboworks.reference import ref_dict


@explicit_serialize
class OptimizeTask(FireTaskBase):
    _fw_name = "OptimizeTask"

    required_params = []
    optional_params = []

#COMPLETED METHODS
    #todo: write docstrings

    def __init__(self, *args, **kwargs):
        super(FireTaskBase, self).__init__(*args, **kwargs)

        #todo: make this work with fw style dictionaries?
        #todo: cleanup attrs + make this not so horrible
        if 'host' in kwargs:
            self._tw_host = kwargs['host']
        else:
            self._tw_host = 'localhost'

        if 'port' in kwargs:
            self._tw_port = kwargs['port']
        else:
            self._tw_port = 27017

        self._tw_port = 27017
        self._tw_host = 'localhost'

        self._tw_mongo = MongoClient(self._tw_host, self._tw_port)
        self._tw_db = self._tw_mongo.turboworks
        self._tw_collection = self._tw_db.turboworks
        self.delimiter = '.'
        self.meta_fw_keys = ['_id', '_tasks']

        self.input_list = []
        self.output_list = []
        self.aux_list = []

        self.tw_spec = {}
        self.extracted=[]

    def run_task(self, fw_spec):
        # This method should be overridden
        pass

    def store(self, fw_spec):

        self.tw_spec = fw_spec
        self._tw_collection.insert_one(OrderedDict(fw_spec))

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

    def auto_extract(self, query = None, label="inputs", n=None):
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
                db.collection.insertOne(my_dict)

            print(auto_extract(['A', 'types.old.q'], label='inputs'))

            [[1200, 12000, 1200], [88, 88, 88]]]

        :param query: (list) of possible (string) ML inputs which will be used as keys searching
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

        if n is None:
            n = self._tw_collection.count()


        if type(query) is not list and query is not None:
            if not isinstance(query[0], string_types):
                raise TypeError("Keys should be in list of strings form. For example, ['X', 'Y.b.z']")


        #todo: change this?
        dict_template = self._tw_collection.find_one()
        query = self.extend_compound_key_list(key_list=query, d=dict_template)

        try:
            query = [x.encode('UTF8') for x in query]
        except:
            pass

        query = sorted(query, key=str.lower)

        if label in ['inputs', 'input', 'in', 'input_list', 'features']:
            self.input_list = query
        elif label in ['outputs', 'output', 'out', 'output_list']:
            self.output_list = query

        for k in query:
            sublist = []

            #todo: check if _tw_collection.find() results in diff order when called
            for doc in self._tw_collection.find():
                if len(sublist) <= n-1:
                    sublist.append(self.extract(k,doc))
            extract.append(sublist)


        # makes single feature/output extracts NOT as a nested list
        # for example, if outputs = ['X'], extract is [32], NOT [[32]]
        if type(query) is list:
            if len(query) == 1:
                extract = extract[0]

        self.extracted = extract
        return extract

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
            keys = sorted(self.input_list, key = str.lower)
        if type(keys) is not list:
            raise TypeError("Keys should be in list form. For example, ['X', 'Y.b.z']")

        if type(updated_values) is list:
            for i, updated_value in enumerate(updated_values):
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

# POSSIBLE IMPROVEMENTS AND TOOLS

    def auto_run(self, inputs=None, outputs=None):
        # automatically runs some default optimization algorithm based on inputs/outputs
        pass


class Utility(object):

    def get_data(wf_func, fw_spec, output_datatypes=None, host='localhost', port=27017):

        """
        Common function for getting data to optimizer implementations
        :param wf_func: fully defined name of the workflow function. This is recommended to be self["func"] as part of
        FireWorks infrastructure

        :param fw_spec: The spec which allows FireWorks to operate.

        :param output_datatypes: The data types which the optimization algorithm can handle as function output. Leave
        blank
        if you do not know what to put here.

        :param host: MongoDB host. Defaults to localhost.

        :param port: MongoDB port. Defaults to local port 27017

        :return: workflow_creator: the fully defined object used to create workflows as part of an optimization loop.

        :return: opt_inputs: all inputs which will be used for the optimization in list of lists form. For example,
        if the function has three parameters and has been run twice, the opt_inputs appears in [run1, run2] form as:

                                opt_inputs = [[1, 20.98, "green"], [4, 23.11, "orange"]]

        :return: opt_outputs: all outputs which will be used for the optimization in list form. For the built in
        algorithms,
        the opt_outputs should only have one output per black box function evaluation. For example, if the black box
        function has been run twice, the opt_outputs appears in [run1, run2] form as:

                                opt_outs = [14.49, 19.34]

        :return: opt_dimensions: dimensions of the search space for the current evaluation. Previous dimensions will be
        stored in the database. The dimensions are in list of tuples/lists form depending on the dimension type. For
        integers and floats, each dimension is (lower, upper). For categorical, each dimension is a list of string
        categories. For example, the dimensions of our example run might be:

                                opt_dimensions = [(1,20), (20.50, 25.00), ["red", "green", "orange", "black"]]

        :return: input_keys: a list of the name of inputs, used for zipping together a dictionary to return to the
        workflow
        creator. This is used if your workflow creator needs the names of the inputs.

        :return: dim_keys: a list of the names of the dimensions, used for zipping together a dict to return to the
        workflow
        This is used if your workflow creator needs the names of the dimensions

        :return: output_keys: a list of the names of the output, used zipping together a dict to return to the workflow.
        This is used if your workflow creator needs the names of the output (it rarely does, unless you have precomputed
        some results. See the tutorial for optimizing new input based on precomputed output for more information).
        """

        # Import only a function named as a module only in the working dir (taken from PyTask)
        toks = wf_func.rsplit(".", 1)
        if len(toks) == 2:
            modname, funcname = toks
            mod = __import__(modname, globals(), locals(), [str(funcname)], 0)
            workflow_creator = getattr(mod, funcname)

        if output_datatypes is None:
            output_datatypes = [np.int64, np.float64, np.int32, np.int64, int, float]

        # Store all spec data in the TurboWorks DB
        mongo = MongoClient(host, port)
        db = mongo.TurboWorks
        collection = db.TurboWorks_collection
        collection.insert_one(OrderedDict(fw_spec))

        # Define optimization variables by reading from DB
        opt_inputs = []
        opt_outputs = []
        opt_dim_history = []
        opt_dimensions = []
        input_keys = []
        dim_keys = []
        output_keys = []
        meta_fw_keys = ['_fw_name', 'func', '_tasks', '_id', '_fw_env']
        cursor = collection.find()

        try:
            basestring
        except NameError:  # Python3 compatibility
            basestring = str

        for document in cursor:
            sublist = []
            subdim = []
            for key in sorted(document['input']):
                if key not in meta_fw_keys:
                    sublist.append(document['input'][key])
                    if key not in input_keys:
                        input_keys.append(key)
            opt_inputs.append(sublist)
            for key in sorted(document['dimensions']):
                if key not in meta_fw_keys:
                    subdim.append(tuple(document['dimensions'][key]))
                    if key not in dim_keys:
                        dim_keys.append(key)
            opt_dim_history.append(subdim)
            for key in document['output']:
                if key not in meta_fw_keys:
                    if key not in output_keys:
                        output_keys.append(key)
                if type(document['output'][key]) in output_datatypes:
                    opt_outputs.append(document['output'][key])
                else:
                    errormsg = 'The optimization must take in a single output. Suported data types are: \n'
                    for datatype in output_datatypes:
                        errormsg += str(datatype) + ' || '

                    errormsg += '\n the given type was ' + str(type(document['output'][key]))
                    raise ValueError(errormsg)

        opt_dimensions = opt_dim_history[-1]

        return workflow_creator, opt_inputs, opt_outputs, opt_dimensions, input_keys, dim_keys, output_keys

@explicit_serialize
class SKOptimizeTask(FireTaskBase):
    """
        This method runs an optimization algorithm for black-box optimization.

        This software uses a modification of the Scikit-Optimize package. Python 3.x is supported.

        :param fw_spec: (dict) specifying the firework data, and the data the firetask will use. The parameters should
        be numerical. The exact layout of this should be defined in the workflow creator.

        :param func: fully defined name of workflow creator function
            example: func="wf_creator.my_wf_creator_file"

        :param min_or_max: decides whether to minimize or maximize the function
            "min": minimizes the function
            "max": maximizes the function
            example: min_or_max= "max"

        :return: FWAction: object which creates new wf object based on updated params and specified workflow creator
        """
    _fw_name = 'SKOptimizeTask'
    required_params = ["func", "min_or_max"]

    def run_task(self, fw_spec):
        """
        The FireTask to be run.
        """

        output_datatypes = [int, float, np.int64, np.float64]

        self.workflow_creator, opt_inputs, opt_outputs, opt_dimensions, input_keys, dim_keys, output_keys = \
            Utility.get_data(self["func"], fw_spec, output_datatypes=output_datatypes, host='localhost', port=27017)

        # Optimization Algorithm and conversion to python native types
        if self["min_or_max"] == "max":
            opt_outputs = [-entry for entry in opt_outputs]
        elif self["min_or_max"] != "min":
            print("TurboWorks: No optima type specified. Defaulting to minimum.")

        new_input = gp_minimize(opt_inputs, opt_outputs, opt_dimensions, acq='EI')

        updated_input = []
        for entry in new_input:
            if type(entry) == np.int64 or type(entry) == int:
                updated_input.append(int(entry))
            elif type(entry) == np.float64 or type(entry) == float:
                updated_input.append(float(entry))
            elif isinstance(entry,basestring) or isinstance(entry,np.unicode_) or isinstance(entry,unicode):
                updated_input.append(str(entry))

        # Create updated dictionary which will be output to the workflow creator
        dim_keys.sort()
        input_keys.sort()
        updated_dictionary = {"input": dict(zip(input_keys, updated_input))}
        current_dimensions = dict(zip(dim_keys, opt_dimensions))
        updated_dictionary["dimensions"] = current_dimensions

        # Initialize new workflow
        return FWAction(additions=self.workflow_creator(updated_dictionary, 'skopt_gp'))

@explicit_serialize
class DummyOptimizeTask(FireTaskBase):
    """
        This method runs a dummy (random sampling) optimization. It works with float, integer, and categorical
        inputs, as well as mixed of those three.

        This software uses a modification of the Scikit-Optimize package. Python 3.x is supported.

        :param fw_spec: (dict) specifying the firework data, and the data the firetask will use. The parameters should
        be numerical. The exact layout of this should be defined in the workflow creator.

        :param func: fully defined name of workflow creator function
            example: func="wf_creator.my_wf_creator_file"

        :return: FWAction: object which creates new wf object based on updated params and specified workflow creator
        """

    _fw_name = 'DummyOptimizeTask'
    required_params = ["func"]

    def run_task(self, fw_spec):
        """
        The FireTask to be run.
        """

        self.workflow_creator, opt_inputs, opt_outputs, opt_dimensions, input_keys, dim_keys, output_keys = \
            Utility.get_data (self["func"], fw_spec, output_datatypes = None, host='localhost', port=27017)

        new_input = dummy_minimize(opt_dimensions)

        updated_input = []
        for entry in new_input:
            if type(entry) == np.int64 or type(entry) == int:
                updated_input.append(int(entry))
            elif type(entry) == np.float64 or type(entry) == float:
                updated_input.append(float(entry))
            elif isinstance(entry, basestring) or isinstance(entry,np.unicode_) or isinstance(entry,unicode):
                updated_input.append(str(entry))

        dim_keys.sort()
        input_keys.sort()
        updated_dictionary = {"input": dict(zip(input_keys, updated_input))}
        current_dimensions = dict(zip(dim_keys, opt_dimensions))
        updated_dictionary["dimensions"] = current_dimensions

        return FWAction(additions=self.workflow_creator(updated_dictionary, 'dummy'))

@explicit_serialize
class COMBOptomizeTask(FireTaskBase):
    """
        This class runs a the Tsudalab COMBO optimization task in a similar fashion to the SKOptimize task. COMBO only
        takes integer input. Duplicate checking is enabled by default.

        :param fw_spec: (dict) specifying the firework data, and the data the firetask will use. The parameters should
        be numerical. The exact layout of this should be defined in the workflow creator.

        :param func: fully defined name of workflow creator function
            example: func="wf_creator.my_wf_creator_file"

        :return: FWAction: object which creates new wf object based on updated params and specified workflow creator
        """
    _fw_name = 'COMBOptimizeTask'
    required_params = ["func","min_or_max"]

    def run_task(self, fw_spec):
        """
        The FireTask to be run.
        """

        self.workflow_creator, opt_inputs, opt_outputs, opt_dimensions, input_keys, dim_keys, output_keys = \
            Utility.get_data(self["func"], fw_spec, output_datatypes=None, host='localhost', port=27017)

        # Optimization Algorithm (with console spam suppressed temporarily)
        '''COMBO's default is maximum, so this is reversed from other optimization task classes.'''

        if self["min_or_max"] == "min":
            opt_outputs = [-entry for entry in opt_outputs]
        elif self["min_or_max"] != "max":
            print("TurboWorks: No optima type specified. Defaulting to maximum.")

        X = calculate_discrete_space(opt_dimensions)
        # X = combo.misc.centering(X)

        def get_input_from_actions(actions, X):
            output = []
            if len(actions)==1:
                output = X[actions[0]]
            else:
                for action in actions:
                    output.append(X[action])
            return output

        def get_actions_from_input(input_list, X):
            actions = []
            for input_vector in input_list:
                actions.append(X.index(tuple(input_vector)))
            return actions

        @contextmanager
        def suppress_stdout():
            with open(os.devnull, "w") as devnull:
                old_stdout = sys.stdout
                sys.stdout = devnull
                try:
                    yield
                finally:
                    sys.stdout = old_stdout

        with suppress_stdout():
            prev_actions = get_actions_from_input(opt_inputs,X)

            policy = combo.search.discrete.policy(test_X=np.asarray(X))
            policy.write(prev_actions, np.asarray(opt_outputs))

            actions = policy.bayes_search(max_num_probes=1, num_search_each_probe=1,
                                          simulator=None, score='EI', interval=0, num_rand_basis=0)
            new_input = list(get_input_from_actions(actions, X))

        # Duplicate protection (this is not dependend on Python native types, numpy comparison is fine)
        new_input = duplicate_check(new_input, opt_inputs, X, 'COMBO')

        # Conversion to Native Types
        updated_input = []
        for entry in new_input:
            if type(entry) == np.int64 or type(entry) == int:
                updated_input.append(int(entry))
            else:
                raise ValueError('The type {} is not supported in COMBO.'.format(type(entry)))

        # Create updated dictionary which will be output to the workflow creator
        dim_keys.sort()
        input_keys.sort()
        updated_dictionary = {"input": dict(zip(input_keys, updated_input))}
        current_dimensions = dict(zip(dim_keys, opt_dimensions))
        updated_dictionary["dimensions"] = current_dimensions

        # Initialize new workflow
        return FWAction(additions=self.workflow_creator(updated_dictionary, 'combo_gp'))

