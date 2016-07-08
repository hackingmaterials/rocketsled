from random import uniform, randint
import numpy as np

def dummy_minimize (my_input, my_output, dimensions):
	if not np.isscalar(my_output[0]):
		raise ValueError(
			"The function to be optimized should return a scalar")
	if np.isscalar(my_input[0]):
		raise ValueError(
			"The function to be optimized should have a list of parameterized inputs")
	new_input=[]

	for i, entry in enumerate(my_input[0]):
		if type(entry) == np.int64 or type(entry) == int:
			lower = dimensions[i][0]
			upper = dimensions[i][1]
			new_param = randint(lower, upper)
			new_input.append(new_param)
		elif type(entry) == np.float64 or type(entry)== float:
			lower = dimensions[i][0]
			upper = dimensions[i][1]
			new_param = uniform(lower, upper)
			new_input.append(new_param)
	return new_input