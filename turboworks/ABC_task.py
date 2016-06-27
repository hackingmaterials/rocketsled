from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase, FWAction
import sys
import numpy as np
# This FireTask is a function A*B/C=D
# It executes this command, checks the parameter range (crude) and saves to TurboworksDB

@explicit_serialize
class ABCtask(FireTaskBase):
	_fw_name = "ABCTask"

	def run_task(self, fw_spec):
		"""
		:param fw_spec: (dict)
		:return: FWAction object which saves all output to spec
		"""
		# Gather inputs from spec
		A_input = fw_spec['A_input']
		B_input = fw_spec['B_input']
		C_input = fw_spec['C_input']

		# Check to make sure params in range, this will need to be replaced with some exception system
		if np.amax([A_input,B_input,C_input]) > 100.00 or np.amin([A_input,B_input,C_input]) < 1:
			sys.exit("One or more parameters is out of range \n A,B, and C must be within 1-100")

		# Run black box objective algorithm (A*B/C = D)
		D_output = A_input*B_input/C_input
		D_write = {'D_output':D_output}

		# Modify changes in spec only
		return FWAction(update_spec=D_write)

