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



To execute this script, change the FW password to today's date (YYYY-MM-DD).
WARNING: This will reset the fireworks database.


"""

# Class for managing FireWorks
launchpad = LaunchPad()

# Class for managing the TurboWorks database directly
manageDB = ManageDB()

# Sample data
A = 92
B = 26
C = 88
my_input = {"A":A, "B":B, "C":C}
dimensions = {"A_range":(1,100),"B_range":(1,100), "C_range":(1,100)}

# Define the initial input dictionary
input_dict = {'input':my_input, 'dimensions':dimensions}

def graph(input_dict, n_runs=30, fw_password=''):
    """
    Execute calculations via FireWorks workflow. Then graph via matplotlib.

    :param: input_dict: the dictionary which will begin the optimization loop

    :param: n_runs: the number of iterations (times a workflow is run) that will be executed for each algorithm.

    :param: fw_password: used for resetting the FireWorks database. To reset it, it should be today's date in the form
    'YYYY-MM-DD'. WARNING: this will erase all previous FireWorks data and will reset all workflows!
    """

    # To reset the workflow, fw_password must be today's date in form 'YYYY-MM-DD'
    launchpad.reset(fw_password, require_password=True)

    # Run n_runs iterations using Skopt Gaussian Processes
    gp_best = []
    wf = workflow_creator(input_dict, 'skopt_gp')
    launchpad.add_wf(wf)
    for i in range(n_runs):
        launch_rocket(launchpad)
        gp_best.append(manageDB.get_optima('D', min_or_max='max')[0])
    manageDB.nuke_it()
    launchpad.defuse_wf(launchpad.get_fw_ids()[-1])

    # Run n_runs iterations using a dummy optimizer (returns random guesses)
    dummy_best = []
    wf = workflow_creator(input_dict, 'dummy')
    launchpad.add_wf(wf)
    for i in range(n_runs):
        launch_rocket(launchpad)
        dummy_best.append(manageDB.get_optima('D', min_or_max='max')[0])
    manageDB.nuke_it()
    launchpad.defuse_wf(launchpad.get_fw_ids()[-1])

    # Run n_runs iterations using COMBO optimization
    combo_best = []
    wf = workflow_creator(input_dict, 'combo_gp')
    launchpad.add_wf(wf)
    for i in range(n_runs):
        launch_rocket(launchpad)
        combo_best.append(manageDB.get_optima('D', min_or_max='max')[0])
    manageDB.nuke_it()

    iterations = list(range(n_runs))
    print("Skopt best:", gp_best[-1])
    print("Dummy best: ", dummy_best[-1])
    print("Combo best:", combo_best[-1])
    skoptline = plt.plot(iterations,gp_best,'g', label = 'skopt')
    dummyline = plt.plot(iterations, dummy_best,'r', label = 'dummy')
    comboline = plt.plot(iterations, combo_best,'b', label = 'combo')
    plt.legend()
    plt.show()

if __name__=="__main__":
    graph(input_dict, n_runs=5, fw_password='2016-08-16')
