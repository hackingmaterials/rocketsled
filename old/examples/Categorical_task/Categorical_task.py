from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase, FWAction
import math

"""
Executes black box function A*B/C
"""

@explicit_serialize
class CategoricalTask(FireTaskBase):
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

        f = fw_spec['input']['food']
        b = fw_spec['input']['beverage']


        # Run black box objective algorithm Branin-Hoo function

        output = 0
        if f=='cookies':
            if b=='milk':
                output = 85
            elif b=='hot chocolate':
                output = 75
            elif b=='water':
                output = 50
            else:
                output = 25
        elif f=='oreos':
            if b == 'milk':
                output = 80
            elif b == 'hot chocolate':
                output = 55
            elif b == 'water':
                output = 30
            else:
                output = 20
        elif f=='strawberries':
            if b=='orange juice':
                output = 60
            else:
                output = 15
        elif f == 'pasta' or f=='burger' or f=='steak':
            if b=='water' or b=='coffee':
                output = 40
            else:
                output = 20
        if b=="beer":
            output+=20
        if b=="milk":
            output+=15

        write2spec = {'output': {'f':output}}
        # Modify changes in spec
        return FWAction(update_spec=write2spec)
