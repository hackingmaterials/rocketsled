#This FireTask will eventually optimize black box algorithms
#Right now it prints a fake optimization for ABCTask
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase, FWAction
from pymongo import MongoClient
import numpy as np
import workflow_creator

@explicit_serialize
class OptimizeTask(FireTaskBase):

   def run_task(self, fw_spec):

#Make sure we are in correct DB
       _fw_name = 'OptimizeTask'
       mongo = MongoClient('localhost', 27017)
       db = mongo.TurboWorks
       collection = db.ABC_collection

#Initialize our variables to be collected from DB
       A_input=[]
       B_input=[]
       C_input=[]
       D_output=[]

#Collect data from DB
       cursor= collection.find({'$or':[{'type':'raw'},{'type':'optimized'}]})
       for document in cursor:
           A_input = A_input + document['A_input']
           B_input = B_input + document['B_input']
           C_input = C_input + document['C_input']
           if "D_output" in document:
               D_output = D_output+document['D_output']

#Fake optimization algorithm, using all previous and new inputs
       A_updated = np.mean(B_input+C_input+D_output)
       A_updated = [A_updated.tolist()]
       B_updated = np.mean(A_input+C_input+D_output)
       B_updated = [B_updated.tolist()]
       C_updated = np.mean(A_input+B_input+D_output)
       C_updated = [C_updated.tolist()]
       if (A_updated>100 or A_updated<1 or B_updated>100
           or B_updated<1 or C_updated>100 or C_updated<1):
           print('\nOptimized parameters will exceed range. Running anyways')

#Convert optmized inputs to dictionary for storage in DB
#these inputs have type = 'optimized'
       # updated_input = {"type":"optimized","A_input":A_updated,"B_input":B_updated,
       #                  "C_input":C_updated}
       # collection.insert_one(updated_input)
       print "\nOptimizeTask ran and determined the updated inputs should be"
       print "  A:", A_updated, "\n  B:", B_updated, "\n  C:", C_updated, "\n"

#Initialize new workflow
       return FWAction(additions=workflow_creator.workflow_creator(A_updated,B_updated,C_updated))
