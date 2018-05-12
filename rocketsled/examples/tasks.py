from __future__ import unicode_literals, print_function, division


"""
Example tasks for use in rs_example optimization workflows.
"""

from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from fireworks import FWAction
import numpy as np


@explicit_serialize
class SumTask(FireTaskBase):
    _fw_name = "SumTask"

    def run_task(self, fw_spec):
        x = fw_spec['_x_opt']
        y = np.sum(x)
        return FWAction(update_spec={'_y_opt': y})

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

@explicit_serialize
class MultiTask2(FireTaskBase):
    _fw_name = "MultiTask2"

    def run_task(self, fw_spec):
        x = fw_spec['_x_opt']
        y = [np.sum(x), np.prod(x)]
        return FWAction(update_spec={'_y_opt': y})

@explicit_serialize
class MultiTask6(FireTaskBase):
    _fw_name = "MultiTask6"

    def run_task(self, fw_spec):
        x = fw_spec['_x_opt']
        y = [np.sum(x), np.prod(x), x[0] ** x[2],
             3.0 * x[1] ** 2 - 4 * x[0] - x[2], max(x), x[0] - x[1] + x[2]]
        return FWAction(update_spec={'_y_opt': y})
