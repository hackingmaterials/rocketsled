from ABC_workflow_creator import workflow_creator
from fireworks.core.rocket_launcher import rapidfire
from fireworks import FWorker, LaunchPad
from fireworks.core.rocket_launcher import launch_rocket
from turboworks.optdb import OptDB
import matplotlib.pyplot as plt

"""
This is the top level script for this turboworks example, and is used primarily for debugging/tuning.
This particular script is used for comparing the effectivness of Skopt and Combo.
"""

launchpad = LaunchPad()
manageDB = OptDB()

# Sample data
A = 45
B = 25
C = 66
input1 = {"A":A, "B":B, "C":C}
dimensions = {"A_range":(1,100),"B_range":(1,100), "C_range":(1,100)}

# Define the initial input
input_dict = {'input':input1, 'dimensions':dimensions}

# How many times to run the workflow + optimization loop
run_num = 30

# Or dynamically call till within a max_val
max_val = 10000
tolerance = .95

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
    manageDB.clean()
    launchpad.defuse_wf(launchpad.get_fw_ids()[-1])

    # Run run_num iterations using a dummy optimizer (returns random)
    dummy_best = []
    wf = workflow_creator(input_dict, 'dummy')
    launchpad.add_wf(wf)
    for i in range(run_num):
        launch_rocket(launchpad)
        dummy_best.append(manageDB.get_optima('D', min_or_max='max')[0])
    manageDB.clean()
    launchpad.defuse_wf(launchpad.get_fw_ids()[-1])

    # Run run_num iterations using COMBO optimization
    combo_best = []
    wf = workflow_creator(input_dict, 'combo_gp')
    launchpad.add_wf(wf)
    for i in range(run_num):
        launch_rocket(launchpad)
        combo_best.append(manageDB.get_optima('D', min_or_max='max')[0])
    manageDB.clean()
    launchpad.defuse_wf(launchpad.get_fw_ids()[-1])

    iterations = list(range(run_num))
    print("GP best:", gp_best[-1])
    print("Dummy best: ", dummy_best[-1])
    print("Combo best:", combo_best[-1])
    plt.plot(iterations,gp_best,'g', iterations, dummy_best,'r', iterations, combo_best,'b')
    # plt.plot(iterations, gp_best, 'g', iterations, dummy_best, 'r')
    plt.show()

def scatter_graph():
    """
    only for dev. graph generation and benchmarking:
        plot the score of each iteration, scatter style
    """

    # Run run_num iterations using Skopt Gaussian Processes
    wf = workflow_creator(input_dict, 'skopt_gp')
    launchpad.add_wf(wf)
    rapidfire(launchpad, FWorker(), nlaunches=run_num, sleep_time=0)
    gp_best = manageDB.get_optima('D', min_or_max='max')[0]
    gp_average = manageDB.get_avg('D')
    gp_total = manageDB.get_param('D')
    manageDB.clean()

    # Run run_num iterations using a dummy optimizer (returns random)
    launchpad.defuse_wf(launchpad.get_fw_ids()[-1])
    wf = workflow_creator(input_dict, 'dummy')
    launchpad.add_wf(wf)
    rapidfire(launchpad, FWorker(), nlaunches=run_num, sleep_time=0)
    dummy_best = manageDB.get_optima('D', min_or_max='max')[0]
    dummy_average = manageDB.get_avg('D')
    dummy_total = manageDB.get_param('D')
    manageDB.clean()

    # Run num_num iterations using COMBO
    launchpad.defuse_wf(launchpad.get_fw_ids()[-1])
    wf = workflow_creator(input_dict, 'combo_gp')
    launchpad.add_wf(wf)
    rapidfire(launchpad, FWorker(), nlaunches=run_num, sleep_time=0)
    combo_best = manageDB.get_optima('D', min_or_max='max')[0]
    combo_average = manageDB.get_avg('D')
    combo_total = manageDB.get_param('D')
    manageDB.clean()
    launchpad.defuse_wf(launchpad.get_fw_ids()[-1])

    # Compare the two optimizations graphically
    print('GP average: ', gp_average, '\n GP best:    ', gp_best)
    print('Dummy average: ', dummy_average, '\n Dummy best: ', dummy_best)
    print('Combo average:', combo_average, '\n Combo best: ', combo_best)
    iterations = list(range(run_num))
    plt.plot(iterations, gp_total, 'g.', iterations, dummy_total, 'r.', iterations, combo_total, '.b')
    plt.show()

def converge_to():
    """
    only for dev. graph generation and benchmarking:
        plot the best result vs the iteration, comparing GP vs random
    """
    # Note: this can take a long time to run, you want to run this outside of Fireworks
    # to get any meaningful graph in your lifetime

    # Run some number of iterations until dummy iteration has converged
    gp_best = []
    gp_iter = 0
    wf = workflow_creator(input_dict, 'skopt_gp')
    launchpad.add_wf(wf)

    gp_iter = gp_iter + 1
    launch_rocket(launchpad)
    gp_best.append(manageDB.get_optima('D', min_or_max='max')[0])

    while (gp_best[-1]<=tolerance*max_val):
        gp_iter = gp_iter+1
        launch_rocket(launchpad)
        gp_best.append(manageDB.get_optima('D', min_or_max='max')[0])

    manageDB.clean()

    # Run some number of iterations until dummy iteration has converged
    launchpad.defuse_wf(launchpad.get_fw_ids()[-1])
    dummy_best = []
    dummy_iter = 0
    wf = workflow_creator(input_dict, 'dummy')
    launchpad.add_wf(wf)
    dummy_iter = dummy_iter + 1
    launch_rocket(launchpad)
    dummy_best.append(manageDB.get_optima('D', min_or_max='max')[0])

    while (dummy_best[-1] <= tolerance * max_val):
        dummy_iter = dummy_iter + 1
        launch_rocket(launchpad)
        dummy_best.append(manageDB.get_optima('D', min_or_max='max')[0])

    manageDB.clean()

    print("GP iterations:", gp_iter)
    print("Dummy iterations:", dummy_iter)

    plt.plot(list(range(gp_iter)), gp_best, 'g', list(range(dummy_iter)), dummy_best, 'r')
    plt.show()

def testing_for_errors():
    """
    only for dev. bugfixing
    """
    wf = workflow_creator(input_dict,'combo_gp')
    launchpad.add_wf(wf)
    rapidfire(launchpad, FWorker(), nlaunches=run_num, sleep_time=0)
    gp_max = manageDB.get_optima('D','max')
    launchpad.defuse_wf(launchpad.get_fw_ids()[-1])
    print (gp_max)

if __name__=="__main__":
    best_graph()