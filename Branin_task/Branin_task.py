from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase, FWAction
import math

"""
Executes black box function A*B/C
"""

@explicit_serialize
class BraninTask(FireTaskBase):
    _fw_name = "BraninTask"

    def run_task(self, fw_spec):
        """
        Executes the Branin-Hoo function

        :param fw_spec: (dict) defines A (float), B (float), and C (float)
        :return: FWAction: object which saves D (float) to the spec
        """
        # Gather inputs from spec
        x1 = fw_spec['input']['x1']
        x2 = fw_spec['input']['x2']

        # Run black box objective algorithm Branin-Hoo function
        pi = 3.14159
        a = 1
        b = 5.1/(4*(pi**2))
        c = 5/pi
        r = 6
        s = 10
        t = 1/(8*pi)
        f = a*((x2 - b*(x1**2)+ c*x1 - r)**2) + s*(1-t)*math.cos(x1) + s
        f_write = {'output': {'f':f}}

        # Modify changes in spec
        return FWAction(update_spec=f_write)
