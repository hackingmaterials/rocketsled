from ABC_workflow_creator import workflow_creator
from fireworks.core.rocket_launcher import rapidfire
from fireworks import FWorker, LaunchPad
from fireworks.core.rocket_launcher import launch_rocket
from manage_DB import count_it, nuke_it, query_it, get_optima, get_avg, get_param

# Set up the launchpad
launchpad = LaunchPad()
launchpad.reset('', require_password=False)

# Sample data
A = 95.0
B = 34.3
C = 88.3
A_dimensions = (1.0, 100.0)
B_dimensions = (1.0, 100.0)
C_dimensions = (1.0, 100.0)

# How many times to automatically run the optimization iteration
run_num = 1000
given = 10000

# Create a workflow and run it
input_dict = {'A_input': A, 'B_input': B, 'C_input': C,
			  'A_dimensions': A_dimensions, 'B_dimensions': B_dimensions, 'C_dimensions': C_dimensions}

wf = workflow_creator(input_dict, 'skopt_gp')
launchpad.add_wf(wf)
rapidfire(launchpad, FWorker(), nlaunches=run_num, sleep_time=0)
gp_best = get_optima('D_output', min_or_max='max')
gp_average = get_avg('D_output')
gp_total = get_param('D_output')
nuke_it()

wf = workflow_creator(input_dict, 'dummy')
launchpad.add_wf(wf)
rapidfire(launchpad, FWorker(), nlaunches=run_num, sleep_time=0)
dummy_best = get_optima('D_output', min_or_max='max')
dummy_average = get_avg('D_output')
dummy_total = get_param('D_output')
nuke_it()

print('GP average: ', gp_average, '\n GP best:    ', gp_best)
print('Dummy average: ', dummy_average, '\n Dummy best: ', dummy_best)

import matplotlib.pyplot as plt
iterations = list(range(run_num))
plt.plot(iterations, gp_total,'g.',iterations, dummy_total,'r.')
plt.show()
