from optimize_task import OptimizeTask
from ABC_task import ABCtask
from fireworks import Firework, Workflow

"""
This file specifies a function that creates a workflow to:
    1. Execute black box function ABCtask
    2. Optimize this function's input parameters
"""

# TODO: fix indentation (mixed tabs and spaces - use spaces). Configure Pycharm so tab automatically maps to spaces
# TODO: install Pylint on Pycharm so you get warnings about these things

# TODO: needs better documentation. e.g. what does input_dict look like?

def workflow_creator(input_dict, opt_method):
	"""
	:param input_dict: dictionary input
	:return: wf: a workflow object describing the above workflow using params entered in input_dict
	"""

	# Assign FireTasks
	firetask1 = ABCtask()
	firetask2 = OptimizeTask(func='ABC_workflow_creator.workflow_creator',
							 opt_method=opt_method, min_or_max="max")

	# Execute FireWork
	fw = [firetask1, firetask2]
	firework1 = Firework(fw, spec=input_dict)
	wf = Workflow([firework1])
	return wf
