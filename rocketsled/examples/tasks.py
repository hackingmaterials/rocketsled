from __future__ import unicode_literals, print_function, division


"""
Example tasks for use in rs_example optimization workflows.
"""

from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from fireworks import FWAction
import numpy as np

__author__ = "Alexander Dunn"
__version__ = "1.0"
__email__ = "ardunn@lbl.gov"


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


@explicit_serialize
class ComplexMultiObjTask(FireTaskBase):
    """
    An example of a complex, multiobjective task with competing objectives.

    This problem is defined on
    """
    _fw_name = "CMOT"

    def run_task(self, fw_spec):
        x = fw_spec['_x_opt']
        fin_len = x[0]
        fin_angle = x[1]
        fin_type = x[2]

        cost = (14.1 * fin_len ** 1.847 + 12.0 + fin_angle * 100.0)/1000.0
        drag = fin_angle ** 0.653 * float(fin_len) ** 1.2
        failure_prob = 0.5 - fin_len/290 + (fin_angle ** 2.0)/16200

        if fin_type == "shark fin":
            cost = cost * 1.05
            drag = drag * 1.15
            failure_prob = failure_prob * 0.75
        elif fin_type == "dolphin fin":
            cost = cost * 1.6
            drag = drag * 0.84
            failure_prob - failure_prob * 1.75

        return FWAction(update_spec={'_y_opt': [cost, drag, failure_prob], '_x_opt': x})