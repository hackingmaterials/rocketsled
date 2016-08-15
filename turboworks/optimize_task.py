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


"""
This FireTask optimizes float, integer, categorical, or mixed float/categorical/integer inputs for black box functions.
"""

@explicit_serialize
class SKOptimizeTask(FireTaskBase):
    _fw_name = 'SKOptimizeTask'
    required_params = ["func", "min_or_max"]

    def run_task(self, fw_spec):
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

        # Import only a function named as a module only in the working dir (taken from PyTask)
        toks = self["func"].rsplit(".", 1)
        if len(toks) == 2:
            modname, funcname = toks
            mod = __import__(modname, globals(), locals(), [str(funcname)], 0)
            self.workflow_creator = getattr(mod, funcname)

        # Store all spec data in DB
        mongo = MongoClient('localhost', 27017)
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
        meta_fw_keys = ['_fw_name', 'func', '_tasks', '_id', '_fw_env']
        cursor = collection.find()

        try:
            basestring
        except NameError:  # Python3 compatibility
            basestring = str

        for document in cursor:
            self.sublist = []
            self.subdim = []
            for key in sorted(document['input']):
                if key not in meta_fw_keys:
                    self.sublist.append(document['input'][key])
                    if key not in input_keys:
                        input_keys.append(key)
            opt_inputs.append(self.sublist)
            for key in sorted(document['dimensions']):
                if key not in meta_fw_keys:
                    self.subdim.append(tuple(document['dimensions'][key]))
                    if key not in dim_keys:
                        dim_keys.append(key)
            opt_dim_history.append(self.subdim)
            for key in document['output']:
                if (type(document['output'][key]) == int or type(document['output'][key]) == float or type(
                        document['output'][key]) == np.int64 or type(document['output'][key]) == np.float64):

                        opt_outputs.append(document['output'][key])
                else:
                    raise ValueError("The optimization algorithm must take in a single output. Supported data types"
                                     "are numpy int64, numpy float64, Python int and Python float")

        opt_dimensions = opt_dim_history[-1]

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

        # Import only a function named as a module only in the working dir (taken from PyTask)
        toks = self["func"].rsplit(".", 1)
        if len(toks) == 2:
            modname, funcname = toks
            mod = __import__(modname, globals(), locals(), [str(funcname)], 0)
            self.workflow_creator = getattr(mod, funcname)

        # Store all spec data in DB
        mongo = MongoClient('localhost', 27017)
        db = mongo.TurboWorks
        collection = db.TurboWorks_collection
        collection.insert_one(OrderedDict(fw_spec))

        opt_dim_history = []
        opt_dimensions = []
        dim_keys = []
        input_keys=[]
        meta_fw_keys = ['_fw_name', 'func', '_tasks', '_id', '_fw_env']
        cursor = collection.find()

        try:
            basestring
        except NameError:  # Python3 compatibility
            basestring = str

        for document in cursor:
            self.subdim = []
            for key in sorted(document['dimensions']):
                if key not in meta_fw_keys:
                    self.subdim.append(tuple(document['dimensions'][key]))
                    if key not in dim_keys:
                        dim_keys.append(key)
            opt_dim_history.append(self.subdim)
            for key in sorted(document['input']):
                if key not in meta_fw_keys:
                    if key not in input_keys:
                        input_keys.append(key)
        opt_dimensions = opt_dim_history[-1]

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
        # Import only a function named as a module only in the working dir (taken from PyTask)
        toks = self["func"].rsplit(".", 1)
        if len(toks) == 2:
            modname, funcname = toks
            mod = __import__(modname, globals(), locals(), [str(funcname)], 0)
            self.workflow_creator = getattr(mod, funcname)

        # Store all spec data in DB
        mongo = MongoClient('localhost', 27017)
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
        meta_fw_keys = ['_fw_name', 'func', '_tasks', '_id', '_fw_env']
        cursor = collection.find()

        try:
            basestring
        except NameError:  # Python3 compatibility
            basestring = str

        for document in cursor:
            self.sublist = []
            self.subdim = []
            for key in sorted(document['input']):
                if key not in meta_fw_keys:
                    self.sublist.append(document['input'][key])
                    if key not in input_keys:
                        input_keys.append(key)
            opt_inputs.append(self.sublist)
            for key in sorted(document['dimensions']):
                if key not in meta_fw_keys:
                    self.subdim.append(tuple(document['dimensions'][key]))
                    if key not in dim_keys:
                        dim_keys.append(key)
            opt_dim_history.append(self.subdim)
            for key in document['output']:
                if (type(document['output'][key]) == int or type(document['output'][key]) == float or type(
                        document['output'][key]) == np.int64 or type(document['output'][key]) == np.float64):

                    opt_outputs.append(document['output'][key])
                else:
                    raise ValueError("The optimization algorithm must take in a single output. Supported data types"
                                     "are numpy int64 and Python int")

        opt_dimensions = opt_dim_history[-1]

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
        new_input = duplicate_check(new_input, opt_inputs, X)

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
