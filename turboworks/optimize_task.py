from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase, FWAction
from pymongo import MongoClient
from gp_opt import gp_minimize
from dummy_opt import dummy_minimize
import numpy as np
from collections import OrderedDict

"""
This FireTask optimizes float, integer, or mixed float/integer inputs for black box functions.
"""


@explicit_serialize

class OptimizeTask(FireTaskBase):
	_fw_name = 'OptimizeTask'
	required_params = ["func", "opt_method", "min_or_max"]

	def run_task(self, fw_spec):
		"""
			This method runs an optimization framework with Fireworks as a FireTask. The algorithms are designed to
			minimize the result by returning an optimal next input. To maximize a result, just make all result data
			negative.

			This software uses a modification of the Scikit-Optimize package. Python 3.x is supported.

        :param fw_spec: (dict) specifying the firework data, and the data the firetask will use. The parameters should
		be numerical in the form of:
			All inputs contain "_input"
				example: "interatomic_spacing_input": 1.4
			All inputs should have dimensions in the form of (upper, lower), and contain "_dimensions"
				example: "interatomic_spacing_dimensions":(0.8, 3.3)
			A single output is labeled with "_output"
				example: "band_gap_output": 3.2

		:param func: fully defined name of workflow creator function
			example: func="wf_creator.my_wf_creator_file"

		:param opt_method: the method of optimization to be used. The current options are
			"skopt_gp": gaussian process optimization using Scikit-Optimize
			"dummy": dummy optimization using completely random sampling in search space
			example: opt_method = "skopt_gp"

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
		keys = []
		meta_fw_keys = ['_fw_name', 'func', '_tasks', '_id', '_fw_env']
		cursor = collection.find()

		for document in cursor:
			self.sublist = []
			self.subdim = []
			for key in sorted(document):
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

		if self["min_or_max"]=="max":
			opt_outputs= [-entry for entry in opt_outputs]
		elif self["min_or_max"]!="min":
			print("TurboWorks: No optima type specified. Defaulting to minimum.")

		# Optimization Algorithm
		if self["opt_method"] == 'skopt_gp':
			new_input = gp_minimize(opt_inputs, opt_outputs, opt_dimensions)
		elif self["opt_method"] == 'dummy':
			new_input = dummy_minimize(opt_dimensions)
		else:
			new_input = opt_inputs[-1]
			print("A valid kwarg is required for the method parameter. Try 'skopt_gp' or 'dummy'.")
			print("Using the most recent inputs as the optimized input.")

		updated_input=[]
		for entry in new_input:
			if type(entry) == np.int64 or type(entry)==int:
				updated_input.append(int(entry))
			elif type(entry) == np.float64 or type(entry)==float:
				updated_input.append(float(entry))

		# Create updated dictionary which will be output to the workflow creator
		input_keys = []
		dim_keys = []
		for key in keys:
			if "_input" in key:
				input_keys.append(key)
			if "_dimension" in key and key not in dim_keys:
				dim_keys.append(key)
		input_keys.sort()
		updated_dictionary = dict(zip(input_keys, updated_input))
		current_dimensions = dict(zip(dim_keys, opt_dimensions))
		total_dict = current_dimensions.copy()
		total_dict.update(updated_dictionary)

		# Initialize new workflow
		return FWAction(additions=self.workflow_creator(total_dict, self["opt_method"]))
