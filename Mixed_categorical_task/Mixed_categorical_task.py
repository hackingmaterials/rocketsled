from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase, FWAction
import math

"""
Executes black box function A*B/C
"""

@explicit_serialize
class MixedCategoricalTask(FireTaskBase):
    _fw_name = "BraninTask"

    def run_task(self, fw_spec):
        """
        Executes the an arbitrary categorical task.

        This function takes in a categorical values for food and beverage and calculates a score based on
        their relationship. The highest scoring combination is 'cookies' and 'milk', which results in a score of 100.

        :param fw_spec: (dict) defines the values of food and beverage
        :return: FWAction: object which saves f('food','beverage') to the spec
        """
        # Gather inputs from spec

        x1 = fw_spec['input']['x1']
        x2 = fw_spec['input']['x2']
        color = fw_spec['input']['color']

        y1 = x1*x2**2

        if color=='red':
            y1 = y1*2
        elif color=='blue':
            y1 = 1.5*y1
        elif color =="green":
            y1 = 0.5*y1
        else:
            y1 = 0.3*y1

        output = y1

        write2spec = {'output': {'f':output}}
        # Modify changes in spec
        return FWAction(update_spec=write2spec)
