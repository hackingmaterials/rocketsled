# SB: I think these comments look better after the import statement
# This FireTask is a function A*B/C=D
# It executes this command, checks the parameter range (crude) and saves to TurboworksDB
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase, FWAction
import sys
import numpy as np
from pymongo import MongoClient

#SB: re-indent the file

@explicit_serialize
class ABCtask(FireTaskBase):
    _fw_name = "ABCTask"

    def run_task(self, fw_spec):
        """
		# SB: Add docstrings for every function in this format, that describe what the function accomplishes, its
		arguments (with types), and return objects/values (with types).
		# SB: Do this for functions in other files too
		:param fw_spec:
        :return:
        """
        # Make sure we are in correct DB
        mongo = MongoClient('localhost', 27017)
        db = mongo.TurboWorks
        collection = db.ABC_collection

        # Gather inputs from spec
        A_input = fw_spec['A_input']
        B_input = fw_spec['B_input']
        C_input = fw_spec['C_input']

        # Check to make sure params in range, this will need to be replaced with some exception system
        if np.amax(A_input + B_input + C_input) > 100.00 or np.amin(A_input + B_input + C_input) < 1:
            sys.exit("One or more parameters is out of range \n A,B, and C must be within 1-100")

        # Run black box objective algorithm (A*B/C = D)
        D_output = np.divide(np.multiply(A_input, B_input), C_input)
        D_output = D_output.tolist()
        print("ABCTask ran correctly. Your D_output is: ", D_output)

        # If there is no updated info, store the values
        if collection.find({'type': 'raw'}).count() == 0:
            ABC_dict = {'type': 'raw', 'A_input': A_input, 'B_input': B_input,
                        'C_input': C_input, 'D_output': D_output}
            collection.insert_one(ABC_dict)

        # We choose not to update the spec, but store everything in the DB
        # This isnt doing anything at the moment
        return FWAction()
