from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase, FWAction

"""
Executes black box function A*B/C
"""


@explicit_serialize
class ABCtask(FireTaskBase):
	_fw_name = "ABCTask"

	def run_task(self, fw_spec):
		"""
		Executes A*B/C

		:param fw_spec: (dict) defines A (float), B (float), and C (float)
		:return: FWAction: object which saves D (float) to the spec
		"""
		# Gather inputs from spec
		A_input = fw_spec['A_input']
		B_input = fw_spec['B_input']
		C_input = fw_spec['C_input']

		# Run black box objective algorithm (A*B/C = D)
		D_output = A_input * B_input / C_input
		D_write = {'D_output': D_output}

		# Modify changes in spec
		return FWAction(update_spec=D_write)
