#This FireTask will eventually optimize black box algorithms
#Right now it prints a fake optimization for ABCTask

import sys
sys.path.append('/Users/alexdunn/Desktop/Project - 1 - Custom Firetask with Arbitrary Optimization')
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase, FWAction
from pymongo import MongoClient

@explicit_serialize
class OptimizeTask(FireTaskBase):

   _fw_name = 'OptimizeTask'
   mongo = MongoClient('localhost', 27017)


   def run_task(self, fw_spec):

#Gather inputs and result of previous Firetask
       A_input = fw_spec['A_input']
       B_input = fw_spec['B_input']
       C_input = fw_spec['C_input']
       D_output = fw_spec['D_output']

       print 'Your optimization algorithm is running using the inputs: \n', A_input, '\n', B_input, '\n', C_input
       print 'and using the outputs: \n', D_output

#Optimization algorithm which maps A,B, and C inputs and a D_output to updated (optimized) inputs
#Right now its just a random assignment
       A_updated = [100.0, 100.0, 100.0, 100.0, 100.0]
       B_updated = [100.0, 100.0, 100.0, 100.0, 100.0]
       C_updated = [0.001, 0.001, 0.001, 0.001, 0.001]


#Print updated spec and modify the current spec
       print 'OptimizeTask ran correctly'
       print 'The optimal inputs for the next iteration are: \n', A_updated, '\n', B_updated, '\n', C_updated
       return FWAction(update_spec={"A_updated": A_updated, "B_updated": B_updated, 'C_updated': C_updated})

#To take updated information, we just get A_input as 'A_updated', etc.
#This way none of the original inputs are overwritten, but subsequent optimized inputs are overwritten
#In the future, if you don't want to store the data in a spec, OptimizeTask would take data from the DB or something