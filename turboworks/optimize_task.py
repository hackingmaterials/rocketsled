"""
FireTask implementation of optimization algorithms.
"""

__author__ = "Alexander Dunn <ardunn@lbl.gov>"
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


#todo: remove this:
from pprint import pprint
from turboworks.reference import ref_dict


@explicit_serialize
class OptimizeTask(FireTaskBase):
    _fw_name = "OptimizeTask"

    required_params = []
    optional_params = []

    def __init__(self, *args, **kwargs):
        super(FireTaskBase, self).__init__(*args, **kwargs)

        #todo: make this work with fw style dictionaries
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


    def run_task(self, fw_spec):
        # This method should be overridden
        pass

    def store(self, fw_spec):
        pprint(fw_spec)
        self._tw_collection.insert_one(OrderedDict(fw_spec))

    def extract(self, k, d):
        '''

        :param k: a single key in class form, e.g. 'types.old.r'
        :param d: a nested dictionary to be searched
        :return: the desired value defined by k
        '''

        if self.delimiter in k:
            keys = k.split(self.delimiter)  # a top down list of nested dictionary keys
        else:
            keys = [k]

        for key in keys:
            try:
                if type(d[key]) == dict and keys.index(key) != (len(keys) - 1):
                    d = d[key]
                else:
                    return d[key]

            except(KeyError):
                pass

    def auto_extract(self, inputs=None, outputs=None, n=None):

        extract = dict()

        for header, ob in {'inputs':inputs, 'outputs':outputs}.iteritems():
            if ob is None:
                continue
            ob.sort()

            extract[header] = []

            for compound_key in ob:
                sublist = []
                for doc in self._tw_collection.find():
                    sublist.append(self.extract(compound_key,doc))
                extract[header].append(sublist)

        return extract

#todo: deprecate to end of file
class HoldingClass(object):

    def gather_single(self, key, type='generic'):

        output = []

        if type=='generic':
            for doc in self._tw_collection.find():
                sublist = []
                for subkey in sorted(doc[key]):
                    sublist.append(doc[key][subkey])

                output.append(sublist)
        elif type=='list':
            for doc in self._tw_collection.find():
                for subkey in sorted(doc[key]):
                    output.append(doc[key][subkey])
        elif type=='dim':
            doc = self._tw_collection.find_one()
            for subkey in sorted(doc[key]):
                output.append(tuple(doc[key][subkey]))

        return output

    def get_keys(self, type='input'):
        keys = []
        print self._tw_cursor

        for doc in self._tw_cursor:
            print(doc)
            # for key in sorted(doc[type]):
            #     keys.append(key) if key not in keys else None

        return keys

    def get_keys_recursive(self):
        pass

    def gather_all(self):
        data = {}
        for key, value in {'input':'generic', 'output':'list', 'dim':'dim'}.items():
            data[key] = self.gather_single(key, type = value)

        return data

    def gather_recursive(self, query, dictionary = None, output=None):
        # recursively searches for query and then gets all instances of it.

        if dictionary is None:
            dict_list = self._tw_cursor
        else:
            dict_list = [dictionary]

        for doc in dict_list:

            if output is None:
                output = []

            for key in doc:
                if key == query:
                    output.append(doc[key])
                elif type(doc[key]) == dict:
                    self.gather_recursive(query=query, dictionary=doc[key], output=output)

        return output

    def gather_all_recursive(self):
        #todo: make this go to lowest variable level and save in form 'upperdict.lowervar'
        pass

    def deconsolidate(self, features=None, matrix=None):
        return dict(zip(features, matrix + [None] * (len(features) - len(matrix))))

    def update_input(self, new_spec, old_spec):
        #automatically associates a recently joined dictionary and puts it back in fw_spec form
        pass


    def create_wf(self, objs = None, storage=None):
        return FWAction(stored_data = storage, additions = objs)


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

