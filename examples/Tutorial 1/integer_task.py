from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase, FWAction

"""
Executes black box function as a FireTask.
"""

@explicit_serialize
class IntegerTask(FireTaskBase):
    _fw_name = "IntegerTask"

    def run_task(self, fw_spec):
        """
        Executes a black box objective task.
        In this case, we are executing A*B/C, with a max score of 1000 and a mean random score of 131.

        :param fw_spec: (dict) defines A (float), B (float), and C (float)
        :return: FWAction: object which saves D (float) to the spec
        """
        # Gather inputs from spec
        A_input = fw_spec['input']['A']
        B_input = fw_spec['input']['B']
        C_input = fw_spec['input']['C']

        # Run black box objective algorithm
        D_output = float(A_input*B_input/C_input)

        # Put the calculated output into a dictionary
        D_write = {'output': {'D':D_output}}

        # Write the changes to the spec
        return FWAction(update_spec=D_write)
