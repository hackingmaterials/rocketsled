from ABC_workflow_creator import workflow_creator
from fireworks.core.rocket_launcher import rapidfire
from fireworks import FWorker, LaunchPad
from fireworks.core.rocket_launcher import launch_rocket
from manage_DB import count_it, nuke_it, query_it, get_optima

# Set up the launchpad
launchpad = LaunchPad()
launchpad.reset('', require_password=False)

# Sample data
A = 42.0
B = 12.0
C = 35.0
A_dimensions = (1.0, 100.0)
B_dimensions = (1.0, 100.0)
C_dimensions = (1.0, 100.0)

# How many times to automatically run the optimization iteration
run_num_array = [1,2,3,4,5,6,7,8,9,10]
iter = len(run_num_array)


# How many times to rerun each entry in run_num_array
reruns = 5

gp_error = []
dummy_error = []
given = 10000

# Create a workflow and run it
input_dict = {'A_input': A, 'B_input': B, 'C_input': C,
			  'A_dimensions': A_dimensions, 'B_dimensions': B_dimensions, 'C_dimensions': C_dimensions}

for i in range(reruns):
	for run_num in run_num_array:

		wf = workflow_creator(input_dict, 'skopt_gp')
		launchpad.add_wf(wf)
		rapidfire(launchpad, FWorker(), nlaunches=run_num, sleep_time=0)
		gp_error.append(given - get_optima('D_output', min_or_max='max'))
		nuke_it()

		wf = workflow_creator(input_dict, 'dummy')
		launchpad.add_wf(wf)
		rapidfire(launchpad, FWorker(), nlaunches=run_num, sleep_time=0)
		dummy_error.append(given-get_optima('D_output', min_or_max='max'))
		nuke_it()

# Plot stuff
import numpy as np
import matplotlib.pyplot as plt

run_num_array_fixed = []

for run_num in run_num_array:
	for i in range(reruns):
		run_num_array_fixed.append(run_num)

run_num_array_fixed.reverse()
gp_error.reverse()
dummy_error.reverse()
for i in range(reruns):
	run_num_array_fixed.append(0)
	gp_error.append(given)
	dummy_error.append(given)
run_num_array_fixed.reverse()
gp_error.reverse()
dummy_error.reverse()

print("Made it!")
print(run_num_array_fixed)
print(gp_error)
print(dummy_error)

y_gp_mean = []
y_dum_mean = []
y_gp_tot =[]
y_dum_tot=[]
y_gp_std = []
y_dum_std=[]
y_gp = gp_error
y_dum = dummy_error
x_iter =[]

x_it = run_num_array_fixed

for i in range(iter):
	j = reruns*i
	x_iter.append(x_it[j])
	y_gp_sub =[]
	y_dum_sub=[]
	for k in range(reruns):
		y_gp_sub.append(y_gp[j+k])
		y_dum_sub.append(y_dum[j+k])
	y_gp_tot.append(y_gp_sub)
	y_dum_tot.append(y_dum_sub)
	y_gp_mean.append(np.mean(y_gp_tot[i]))
	y_dum_mean.append(np.mean(y_dum_tot[i]))
	y_gp_std.append(np.std(y_gp_tot[i]))
	y_dum_std.append(np.std(y_gp_tot[i]))

print("\n---------------transformed-----------------")
print(x_iter)
print(y_gp_mean)
print(y_dum_mean)

plt.plot(x_iter, y_gp_mean,'g', x_iter, y_dum_mean, 'r')
plt.fill_between(x_iter, np.add(y_gp_mean,y_gp_std), np.subtract(y_gp_mean,y_gp_std), color='green', alpha=0.2)
plt.fill_between(x_iter, np.add(y_dum_mean,y_dum_std), np.subtract(y_dum_mean, y_dum_std), color='red', alpha=0.2)
plt.xlabel("Number of iterations")
plt.ylabel("Deviation from optimal value")
plt.show()
