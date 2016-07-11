from ABC_workflow_creator import workflow_creator
from fireworks.core.rocket_launcher import rapidfire
from fireworks import FWorker, LaunchPad
from fireworks.core.rocket_launcher import launch_rocket
from turboworks.manage_DB import ManageDB
import matplotlib.pyplot as plt

"""
This is the top level script for this turboworks example.
"""

launchpad = LaunchPad()
manageDB = ManageDB()

# Sample data
A = 45.1
B = 22.3
C = 67.0
A_dimensions = (1.0, 100.0)
B_dimensions = (1.0, 100.0)
C_dimensions = (1, 100)

# How many times to run the workflow + optimization loop
run_num = 1

# Define the initial input
input_dict = {'A_input': A, 'B_input': B, 'C_input': C,
			  'A_dimensions': A_dimensions, 'B_dimensions': B_dimensions, 'C_dimensions': C_dimensions}

def test_opt_best_vs_iter():
    """
    only for dev. graph generation and benchmarking:
        plot the best result vs the iteration, comparing GP vs random
    """

    # Run run_num iterations using Skopt Gaussian Processes
    gp_best = []
    launchpad.reset('2016-07-11')
    wf = workflow_creator(input_dict, 'skopt_gp')
    launchpad.add_wf(wf)
    for i in range(run_num):
        launch_rocket(launchpad)
        gp_best.append(manageDB.get_optima('D_output', min_or_max='max'))
    manageDB.nuke_it()

    # Run run_num iterations using a dummy optimizer (returns random)
    dummy_best = []
    launchpad.reset('2016-07-11')
    wf = workflow_creator(input_dict, 'dummy')
    launchpad.add_wf(wf)
    for i in range(run_num):
        launch_rocket(launchpad)
        dummy_best.append(manageDB.get_optima('D_output', min_or_max='max'))
    manageDB.nuke_it()

    iterations = list(range(run_num))
    plt.plot(iterations,gp_best,'g', iterations, dummy_best,'r')
    plt.show()

def test_opt_every_point():
    """
    only for dev. graph generation and benchmarking:
        plot the score of each iteration, scatter style
    """

    # Run run_num iterations using Skopt Gaussian Processes
    wf = workflow_creator(input_dict, 'skopt_gp')
    launchpad.add_wf(wf)
    rapidfire(launchpad, FWorker(), nlaunches=run_num, sleep_time=0)
    gp_best = manageDB.get_optima('D_output', min_or_max='max')
    gp_average = manageDB.get_avg('D_output')
    gp_total = manageDB.get_param('D_output')
    manageDB.nuke_it()

    # Run run_num iterations using a dummy optimizer (returns random)
    launchpad.reset('', require_password=False)
    wf = workflow_creator(input_dict, 'dummy')
    launchpad.add_wf(wf)
    rapidfire(launchpad, FWorker(), nlaunches=run_num, sleep_time=0)
    dummy_best = manageDB.get_optima('D_output', min_or_max='max')
    dummy_average = manageDB.get_avg('D_output')
    dummy_total = manageDB.get_param('D_output')
    manageDB.nuke_it()

    # Compare the two optimizations graphically
    print('GP average: ', gp_average, '\n GP best:    ', gp_best)
    print('Dummy average: ', dummy_average, '\n Dummy best: ', dummy_best)
    iterations = list(range(run_num))
    plt.plot(iterations, gp_total, 'g.', iterations, dummy_total, 'r.')
    plt.show()

if __name__=="__main__":
    test_opt_every_point()