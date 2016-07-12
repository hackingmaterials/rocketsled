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
A = 4.1
B = 3.3
C = 85.3
input1 = {"A":A, "B":B, "C":C}
dimensions = {"A_range":(1.0,100.0),"B_range":(1.0,100.0), "C_range":(1.0,100.0)}

# Define the initial input
input_dict = {'input':input1, 'dimensions':dimensions}



# How many times to run the workflow + optimization loop
run_num = 50


def best_graph():
    """
    only for dev. graph generation and benchmarking:
        plot the best result vs the iteration, comparing GP vs random
    """

    # Run run_num iterations using Skopt Gaussian Processes
    gp_best = []
    launchpad.reset('2016-07-12')
    wf = workflow_creator(input_dict, 'skopt_gp')
    launchpad.add_wf(wf)
    for i in range(run_num):
        launch_rocket(launchpad)
        gp_best.append(manageDB.get_optima('D', min_or_max='max')[0])
    manageDB.nuke_it()

    # Run run_num iterations using a dummy optimizer (returns random)
    dummy_best = []
    launchpad.reset('2016-07-12')
    wf = workflow_creator(input_dict, 'dummy')
    launchpad.add_wf(wf)
    for i in range(run_num):
        launch_rocket(launchpad)
        dummy_best.append(manageDB.get_optima('D', min_or_max='max')[0])
    manageDB.nuke_it()

    iterations = list(range(run_num))
    plt.plot(iterations,gp_best,'g', iterations, dummy_best,'r')
    plt.show()

def scatter_graph():
    """
    only for dev. graph generation and benchmarking:
        plot the score of each iteration, scatter style
    """

    # Run run_num iterations using Skopt Gaussian Processes
    launchpad.reset('', require_password=False)
    wf = workflow_creator(input_dict, 'skopt_gp')
    launchpad.add_wf(wf)
    rapidfire(launchpad, FWorker(), nlaunches=run_num, sleep_time=0)
    gp_best = manageDB.get_optima('D', min_or_max='max')[0]
    gp_average = manageDB.get_avg('D')
    gp_total = manageDB.get_param('D')
    manageDB.nuke_it()

    # Run run_num iterations using a dummy optimizer (returns random)
    launchpad.reset('', require_password=False)
    wf = workflow_creator(input_dict, 'dummy')
    launchpad.add_wf(wf)
    rapidfire(launchpad, FWorker(), nlaunches=run_num, sleep_time=0)
    dummy_best = manageDB.get_optima('D', min_or_max='max')[0]
    dummy_average = manageDB.get_avg('D')
    dummy_total = manageDB.get_param('D')
    manageDB.nuke_it()

    # Compare the two optimizations graphically
    print('GP average: ', gp_average, '\n GP best:    ', gp_best)
    print('Dummy average: ', dummy_average, '\n Dummy best: ', dummy_best)
    iterations = list(range(run_num))
    plt.plot(iterations, gp_total, 'g.', iterations, dummy_total, 'r.')
    plt.show()

def testing_for_errors():
    """
    only for dev. bugfixing
    """
    launchpad.reset('', require_password=False)
    wf = workflow_creator(input_dict,'skopt_gp')
    launchpad.add_wf(wf)
    rapidfire(launchpad, FWorker(), nlaunches=run_num, sleep_time=0)
    gp_max = manageDB.get_optima('D','max')
    print(gp_max)

if __name__=="__main__":
    testing_for_errors()