"""
Example tasks for use in example optimization workflows.
"""

from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from fireworks import FWAction


@explicit_serialize
class BasicCalculateTask(FireTaskBase):
    _fw_name = "BasicCalculateTask"

    def run_task(self, fw_spec):

        A = fw_spec['A']
        B = fw_spec['B']
        C = fw_spec['C']

        score = A*B/C
        return FWAction(update_spec={'_y_opt': score})

@explicit_serialize
class MixedCalculateTask(FireTaskBase):
    _fw_name = "MixedCalculateTask"

    def run_task(self, fw_spec):

        A = fw_spec['A']
        B = fw_spec['B']
        C = fw_spec['C']
        D = fw_spec['D']

        score = A**2 + B**2 / C
        score += 30 if D == 'red' else 0
        score -= 20 if D == 'green' else 0

        return FWAction(update_spec={'_y_opt': score})
