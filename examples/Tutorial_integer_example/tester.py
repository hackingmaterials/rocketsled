import numpy as np
# output_datatypes = [np.int64, np.int32, np.float64, int, float]
#
# errormsg = 'The optimization must take in a single output. Suported data types are: \n'
# for datatype in output_datatypes:
#     errormsg += str(datatype) + ' || '
# raise ValueError(errormsg)


output_datatypes = [int, float, np.int64, np.float64]
print type(output_datatypes)
if type(output_datatypes) == list:
    print "passed"