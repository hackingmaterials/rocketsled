from integer_task_workflow_creator import workflow_creator
from fireworks.core.rocket_launcher import rapidfire
from fireworks import FWorker, LaunchPad
from fireworks.core.rocket_launcher import launch_rocket
from turboworks.manage_DB import ManageDB
import matplotlib.pyplot as plt

"""
This is the tutorial script for an integer task in TurboWorks.

Here we compare the effectiveness of Skopt, COMBO, and random guessing on optimizing an arbitrary black box function:

        A*B/C = D

Where A,B, and C are parameters which range from 1-100, and
we would like to find the parameters which result in the maximum value of D.
"""

launchpad = LaunchPad()
manageDB = ManageDB()

# Sample data
A = 12
B = 26
C = 88
my_input = {"A":A, "B":B, "C":C}
dimensions = {"A_range":(1,100),"B_range":(1,100), "C_range":(1,100)}

# Define the initial input
input_dict = {'input':my_input, 'dimensions':dimensions}

# How many times to run the workflow + optimization loop
run_num = 10

launchpad.reset('', require_password=False)

def best_graph():
    """
    only for dev. graph generation and benchmarking:
        plot the best result vs the iteration, comparing GP vs random
    """

    # Run run_num iterations using Skopt Gaussian Processes
    gp_best = []
    wf = workflow_creator(input_dict, 'skopt_gp')
    launchpad.add_wf(wf)
    for i in range(run_num):
        launch_rocket(launchpad)
        gp_best.append(manageDB.get_optima('D', min_or_max='max')[0])
    manageDB.nuke_it()
    launchpad.defuse_wf(launchpad.get_fw_ids()[-1])

    # Run run_num iterations using a dummy optimizer (returns random)
    dummy_best = []
    wf = workflow_creator(input_dict, 'dummy')
    launchpad.add_wf(wf)
    for i in range(run_num):
        launch_rocket(launchpad)
        dummy_best.append(manageDB.get_optima('D', min_or_max='max')[0])
    manageDB.nuke_it()
    launchpad.defuse_wf(launchpad.get_fw_ids()[-1])

    # Run run_num iterations using COMBO optimization
    combo_best = []
    wf = workflow_creator(input_dict, 'combo_gp')
    launchpad.add_wf(wf)
    for i in range(run_num):
        launch_rocket(launchpad)
        combo_best.append(manageDB.get_optima('D', min_or_max='max')[0])
    manageDB.nuke_it()

    iterations = list(range(run_num))
    print("GP best:", gp_best[-1])
    print("Dummy best: ", dummy_best[-1])
    print("Combo best:", combo_best[-1])
    plt.plot(iterations,gp_best,'g', iterations, dummy_best,'r', iterations, combo_best,'b')
    plt.plot(iterations, combo_best,'b')
    plt.show()


if __name__=="__main__":
    best_graph()