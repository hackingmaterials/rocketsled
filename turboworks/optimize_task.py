#This FireTask will eventually optimize black box algorithms
#Right now it prints a fake optimization for ABCTask
#Right now it can take arrays of any size and JSON files with 3-4 integer inputs and a string specifying the type for the
#file. In the useful cases, the type = 'data'

import sys
sys.path.append('/Users/alexdunn/Desktop/Project - 1 - Custom Firetask with Arbitrary Optimization')
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase, FWAction
from pymongo import MongoClient
import numpy as np

@explicit_serialize
class OptimizeTask(FireTaskBase):

   def run_task(self, fw_spec):

#Connect to DB
       _fw_name = 'OptimizeTask'
       mongo = MongoClient('localhost', 27017)
       db = mongo.TurboWorks
       collection = db.ABC_collection

#Initialize our variables to be collected from DB
       A_input=[]
       B_input=[]
       C_input=[]
       D_output=[]


#Fake optimization algorithm (FAO)
    #FAO maps all A,B, and C inputs AND D output from all workflows to some made up optimization function which is:
    #optimum A = mean(all elements of A,B,C, and D) + mean(A)
    #optimim B = mean(all elements of A,B,C, and D) + mean(B)
    #optimum C = mean(all elements of A,B,C, and D) + mean(C)

       cursor = collection.find({'type':'data'})
       for document in cursor:
           A_input = A_input + document['A_input']
           B_input = B_input + document['B_input']
           C_input = C_input + document['C_input']
           if "D_output" in document:
               D_output = D_output+document['D_output']

       A_updated = np.mean(A_input+B_input+C_input+D_output)+np.mean(A_input)
       A_updated = A_updated.tolist()
       B_updated = np.mean(A_input+B_input+C_input+D_output)+np.mean(B_input)
       B_updated = B_updated.tolist()
       C_updated = np.mean(A_input+B_input+C_input+D_output)+np.mean(C_input)
       C_updated = C_updated.tolist()

#Convert to dictionary for storage in DB
       updated_input = {"A_updated":A_updated,"B_updated":B_updated,"C_updated":C_updated}
       collection.insert_one(updated_input)
       print "\nOptimizeTask ran and determined the updated inputs should be"
       print "  A:", A_updated, "\n  B:", B_updated, "\n  C:", C_updated, "\n"

#Initialize new workflow
       # return FWAction(update_spec={"A_updated": A_updated, "B_updated": B_updated, 'C_updated': C_updated})


#ITS CREATING TWO DOCUMENTS BECAUSE ITS SAVING THE UPDATED ONE AS SEPARATE FROM THE INITIAL
#CANT DELETE ANYTHING FROM PYMONGO?
#MUST CREATE NEW SPEC FOR NEW WORKFLOW WHEN FWACTION RETURNED