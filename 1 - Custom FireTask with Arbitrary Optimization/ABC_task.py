#This FireTask is a function A*B/C=D
#It executes this command, saves the output in the DB, and updates the same spec

import sys
sys.path.append('/Users/alexdunn/Desktop/')
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase, FWAction
import numpy as np

@explicit_serialize

class ABCtask(FireTaskBase):
   _fw_name = "ABCTask"

   def run_task(self, fw_spec):

#Gather inputs from spec
       A_input = fw_spec['A_input']
       B_input = fw_spec['B_input']
       C_input = fw_spec['C_input']


#Run black box objective algorithm (A*B/C = D)
       D_output = np.divide(np.multiply(A_input, B_input), C_input)
       print("ABCTask ran correctly. Your D_output is: ", D_output)

#Update spec with result, AND store data in DB
       return FWAction(stored_data={'D_output':D_output}, update_spec={"D_output":D_output})
