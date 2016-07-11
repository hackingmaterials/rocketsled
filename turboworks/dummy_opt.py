from random import uniform, randint
import numpy as np


# TODO: fix indentation (mixed tabs and spaces - use spaces). Configure Pycharm so tab automatically maps to spaces
# TODO: install Pylint on Pycharm so you get warnings about these things

def dummy_minimize(dimensions):
	"""
	This function returns random new inputs based on the dimensions of the search space.
	It works with float and integer types.

	:param dimensions (list of 2-tuples): defines the dimensions of each parameter
	:return: new_input (list): randomly chosen next parameters in the search space
	"""

	new_input = []
	for i, (lower, upper) in enumerate(dimensions):
		if type(upper) == np.int64 or type(upper) == int:
			new_param = randint(lower, upper)
			new_input.append(new_param)
		elif type(upper) == np.float64 or type(upper) == float:
			new_param = uniform(lower, upper)
			new_input.append(new_param)
	return new_input
