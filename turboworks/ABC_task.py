#This FireTask is a function A*B/C=D
#It executes this command, saves the output in the DB, and updates the same spec

import sys
sys.path.append('/Users/alexdunn/Desktop/')
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase, FWAction
import numpy as np
from pymongo import MongoClient

@explicit_serialize

class ABCtask(FireTaskBase):
   _fw_name = "ABCTask"

   def run_task(self, fw_spec):

#Set up mongo DB
       mongo = MongoClient('localhost', 27017)
       db = mongo.TurboWorks
       collection = db.ABC_collection

#Gather inputs from spec
       A_input = fw_spec['A_input']
       B_input = fw_spec['B_input']
       C_input = fw_spec['C_input']

#Run black box objective algorithm (A*B/C = D)
       D_output = np.divide(np.multiply(A_input, B_input), C_input)
       D_output = D_output.tolist()
       print("ABCTask ran correctly. Your D_output is: ", D_output)

#Store all input and output as dict and store in custom TurboWorks DB
       ABC_dict = {'type':'data','A_input':A_input, 'B_input':B_input,'C_input':C_input,'D_output':D_output}
       collection.insert_one(ABC_dict)

# #Ensure data got to db ok
#        cursor = collection.find()
#        for document in cursor:
#            print document
#            print 'Next doc'

#Update spec with result, AND store data in DB
       return FWAction(update_spec={"D_output":D_output})
