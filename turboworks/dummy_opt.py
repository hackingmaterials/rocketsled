from random import uniform
import numpy as np

def dummy_minimize (my_input, my_output, dimensions):
	if not np.isscalar(my_output[0]):
		raise ValueError(
			"The function to be optimized should return a scalar")
	if np.isscalar(my_input[0]):
		raise ValueError(
			"The function to be optimized should have a list of parameterized inputs")
	new_input=[]
	num_params = len(my_input[0])

	print (num_params)
	for i in range(num_params):
		lower = dimensions[i][0]
		upper = dimensions[i][1]
		rand_in_range = uniform(lower, upper)
		new_input.append(rand_in_range)

	return new_input