from ABC_workflow_creator import workflow_creator
from fireworks.core.rocket_launcher import rapidfire
from fireworks import FWorker, LaunchPad
from fireworks.core.rocket_launcher import launch_rocket
from manage_DB import count_it, nuke_it, query_it, get_optima, get_avg, get_param
import matplotlib.pyplot as plt

# Set up the launchpad
launchpad = LaunchPad()
launchpad.reset('', require_password=False)

# Sample data
A = 45.1
B = 22.3
C = 67.0
A_dimensions = (1.0, 100.0)
B_dimensions = (1.0, 100.0)
C_dimensions = (1, 100)

# How many times to run the workflow + optimization loop
# 1-5 usually takes <10s, 100 takes 5min, 1000+ will take an hour+
run_num = 200

# Define the initial input
input_dict = {'A_input': A, 'B_input': B, 'C_input': C,
			  'A_dimensions': A_dimensions, 'B_dimensions': B_dimensions, 'C_dimensions': C_dimensions}

# Run run_num iterations using Skopt Gaussian Processes
wf = workflow_creator(input_dict, 'skopt_gp')
launchpad.add_wf(wf)
rapidfire(launchpad, FWorker(), nlaunches=run_num, sleep_time=0)
gp_best = get_optima('D_output', min_or_max='max')
gp_average = get_avg('D_output')
gp_total = get_param('D_output')
nuke_it()

# Run run_num iterations using a dummy optimizer (returns random)
launchpad.reset('2016-07-08')
wf = workflow_creator(input_dict, 'dummy')
launchpad.add_wf(wf)
rapidfire(launchpad, FWorker(), nlaunches=run_num, sleep_time=0)
dummy_best = get_optima('D_output', min_or_max='max')
dummy_average = get_avg('D_output')
dummy_total = get_param('D_output')
nuke_it()

# Compare the two optimizations grpahically
print('GP average: ', gp_average, '\n GP best:    ', gp_best)
print('Dummy average: ', dummy_average, '\n Dummy best: ', dummy_best)
iterations = list(range(run_num))
plt.plot(iterations, gp_total, 'g.', iterations, dummy_total, 'r.')
plt.show()
