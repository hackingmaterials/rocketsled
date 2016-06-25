from optimize_task import OptimizeTask
from ABC_task import ABCtask
from fireworks import Firework, Workflow

# This file specifies the workflow

def workflow_creator(A_input, B_input, C_input):

	# Assign FireTasks
	firetask1 = ABCtask()
	# firetask2 = OptimizeTask()
	firetask2 = OptimizeTask('ABC_workflow_creator')

	# Execute FireWork
	fw = [firetask1, firetask2]
	firework1 = Firework(fw, spec={"A_input": A_input, "B_input": B_input,
								   "C_input": C_input})
	wf = Workflow([firework1])
	return wf
