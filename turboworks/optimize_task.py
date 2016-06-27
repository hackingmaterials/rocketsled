from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase, FWAction
from pymongo import MongoClient
import numpy as np
import pprint
import copy

# This FireTask will eventually optimize black box algorithms
# Right now it prints a fake optimization for ABCTask

@explicit_serialize
class OptimizeTask(FireTaskBase):

	_fw_name = 'OptimizeTask'
	required_params = ["func"]
	optional_params = ['args','kwargs']

	def run_task(self, fw_spec):
		"""
		:param fw_spec: (dict)
		:return: FWAction object which creates new wf object
		"""
		# Import only a function named as a module only in the working dir
		toks = self["func"].rsplit(".", 1)
		if len(toks) == 2:
			modname, funcname = toks
			mod = __import__(modname, globals(), locals(), [str(funcname)], 0)
			self.workflow_creator= getattr(mod, funcname)

		# Make sure we are in correct DB
		mongo = MongoClient('localhost', 27017)
		db = mongo.TurboWorks
		collection = db.ABC_collection

		# Store spec data in DB
		collection.insert_one(fw_spec)

		# Read all DB data
		keys=[]
		vals=[]
		original_dict={}
		cursor = collection.find()
		meta_fw_keys = ['_fw_name', 'func', '_tasks', '_id', '_fw_env']
		for document in cursor:
			temp_keys = document.keys()
			for key in temp_keys:
				if key not in meta_fw_keys:
					if key not in keys:
						keys.append(key)
					if key in original_dict:
						original_dict[key] = original_dict[key] + [document[key]]
					else:
						original_dict[key] = [document[key]]

		print '\nOptimize task will be running using the following data:'
		pprint.pprint(original_dict)

		# Fake optimization algorithm
		# Determines the updated values to guess
		updated_dict = {}
		sums = []
		for key in original_dict:
			if 'input' in key:
				sums = sums + original_dict[key]
				updated_dict[key] = np.multiply(0.25*np.random.rand()+0.875,np.mean(sums))
		print '\nOptimize task ran to completion.\nThe following inputs are the optimal next inputs:'
		pprint.pprint(updated_dict)

		# Initialize new workflow
		return FWAction(additions=self.workflow_creator(updated_dict))

