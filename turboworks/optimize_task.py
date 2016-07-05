from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase, FWAction
from pymongo import MongoClient
from skopt.gp_opt import gp_minimize
from collections import OrderedDict
from pprint import pprint

"""
This FireTask optimizes inputs for black box functions.
"""

@explicit_serialize
class OptimizeTask(FireTaskBase):

	_fw_name = 'OptimizeTask'
	required_params = ["func"]
	optional_params = ['args','kwargs']

	def run_task(self, fw_spec):
		"""
		This method gets called when the FireTask is run. It can take in a
        Firework spec, perform some task using that data, and then return an
        output in the form of a FWAction.
		:param fw_spec: (dict) specifying the firework data, and the data the firetask will use
		:return: FWAction: object which creates new wf object based on updated params and specified workflow creator
		"""
		# Import only a function named as a module only in the working dir (taken from PyTask)
		toks = self["func"].rsplit(".", 1)
		if len(toks) == 2:
			modname, funcname = toks
			mod = __import__(modname, globals(), locals(), [str(funcname)], 0)
			self.workflow_creator= getattr(mod, funcname)

		# Make sure we are in correct DB
		mongo = MongoClient('localhost', 27017)
		db = mongo.TurboWorks
		collection = db.TurboWorks_collection

		# Store all spec data in DB
		collection.insert_one(OrderedDict(fw_spec))

		# Define our optimization variables by reading from DB
		opt_inputs = []
		opt_outputs = []
		opt_dim_history = []
		opt_dimensions = []
		keys = []
		meta_fw_keys = ['_fw_name', 'func', '_tasks', '_id', '_fw_env', 'api']
		cursor = collection.find()

		for document in cursor:
			self.sublist = []
			self.subdim = []
			for key in sorted(document):
				# print(key)
				if key not in meta_fw_keys:
					if key not in keys:
						keys.append(key)
					if '_input' in key:
						self.sublist.append(document[key])
					if '_dimensions' in key:
						self.subdim.append(tuple(document[key]))
					if '_output' in key:
						opt_outputs.append(document[key])
			opt_inputs.append(self.sublist)
			opt_dim_history.append(self.subdim)
		opt_dimensions = opt_dim_history[-1]

		# Optimization Algorithm
		new_input = gp_minimize(opt_inputs,opt_outputs,opt_dimensions,
									maxiter=2, n_start=1)
		updated_input = [float(entry) for entry in new_input]

		# Create dictionary which will be output to the workflow creator
		input_keys = []
		dim_keys = []
		for key in keys:
			if "_input" in key:
				input_keys.append(key)
			if "_dimension" in key and key not in dim_keys:
				dim_keys.append(key)
		input_keys.sort()
		updated_dictionary = dict(zip(input_keys,updated_input))
		current_dimensions = dict(zip(dim_keys,opt_dimensions))
		total_dict = current_dimensions.copy()
		total_dict.update(updated_dictionary)

		pprint(updated_dictionary)

		# Initialize new workflow
		return FWAction(additions=self.workflow_creator(total_dict))

