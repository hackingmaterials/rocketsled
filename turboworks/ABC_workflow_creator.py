from optimize_task import OptimizeTask
from ABC_task import ABCtask
from fireworks import Firework, Workflow

# This file specifies the workflow

def workflow_creator(input_dict):

	# Assign FireTasks
	firetask1 = ABCtask()
	# firetask2 = OptimizeTask()
	firetask2 = OptimizeTask(func='ABC_workflow_creator.workflow_creator')

	# Execute FireWork
	fw = [firetask1, firetask2]
	firework1 = Firework(fw, spec=input_dict)
	wf = Workflow([firework1])
	return wf
