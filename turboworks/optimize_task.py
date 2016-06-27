from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase, FWAction
from pymongo import MongoClient
import importlib
import numpy as np
# import ABC_workflow_creator

# This FireTask will eventually optimize black box algorithms
# Right now it prints a fake optimization for ABCTask

@explicit_serialize
class OptimizeTask(FireTaskBase):

	#PROBLEM AREA: check test_code.py and whats being imported to see whats happening
	# def __init__(self, wf_module_name):
	# 	print '\n \n \n--------------------------------------'
	# 	print 'WHATS BEING IMPORTED:', wf_module_name
	# 	print '-------------------------------------'
	# 	wf = importlib.import_module(wf_module_name)
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

		# Gather recent data from spec
		A_read = fw_spec['A_input']
		B_read = fw_spec['B_input']
		C_read = fw_spec['C_input']
		D_read = fw_spec['D_output']

		# Store all data in DB
		ABCD_write = {'A_input': A_read, 'B_input': B_read,
					  'C_input': C_read, 'D_output': D_read}
		collection.insert_one(ABCD_write)

		# Append DB data to spec data
		A_input = []
		B_input = []
		C_input = []
		D_output = []
		cursor = collection.find()
		for document in cursor:
			A_input = A_input + [document['A_input']]
			B_input = B_input + [document['B_input']]
			C_input = C_input + [document['C_input']]
			D_output = D_output + [document['D_output']]

		# Fake optimization algorithm, using all previous and new inputs
		A_updated = 1.05 * (np.mean(A_input + B_input + C_input))
		B_updated = 1.01 * (np.mean(A_input + B_input + C_input))
		C_updated = .95 * (np.mean(A_input + B_input + C_input))

		if (A_updated > 100 or A_updated < 1 or B_updated > 100
			or B_updated < 1 or C_updated > 100 or C_updated < 1):
			print('\nOptimized parameters will exceed range. Running anyways')
		print "\nOptimizeTask ran and determined the updated inputs should be"
		print "  A:", A_updated, "\n  B:", B_updated, "\n  C:", C_updated, "\n"

		# Initialize new workflow
		return FWAction(additions=self.workflow_creator(A_updated, B_updated, C_updated))
		# return FWAction(additions=ABC_workflow_creator.workflow_creator(A_updated,B_updated,C_updated))
