from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase, FWAction

"""
Executes black box function
"""

@explicit_serialize
class ABCtask(FireTaskBase):
    _fw_name = "ABCTask"

    def run_task(self, fw_spec):
        """
        Executes

        :param fw_spec: (dict) defines A (float), B (float), and C (float)
        :return: FWAction: object which saves D (float) to the spec
        """
        # Gather inputs from spec
        A_input = fw_spec['input']['A']
        B_input = fw_spec['input']['B']
        C_input = fw_spec['input']['C']

        # Run black box objective algorithm

        D_output = A_input**2/100 - 3*B_input + C_input
        if B_input > 45 and B_input < 55:
            D_output +=10
            if B_input == 50:
                D_output += 10
        if A_input > 45 and A_input < 55:
            D_output += 10
            if A_input == 50:
                D_output += 10
        if C_input < 10:
            D_output += 10
            if C_input == 5:
                D_output +=10


        D_write = {'output': {'D':D_output}}

        # Modify changes in spec
        return FWAction(update_spec=D_write)
