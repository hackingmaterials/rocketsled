from Categorical_workflow_creator import workflow_creator
from fireworks.core.rocket_launcher import rapidfire
from fireworks import FWorker, LaunchPad
from fireworks.core.rocket_launcher import launch_rocket
from turboworks.manage_db import ManageDB
import matplotlib.pyplot as plt

"""
This is the top level script for this turboworks example, and is used primarily for debugging/tuning.
"""

launchpad = LaunchPad()
manageDB = ManageDB()

# Sample data
my_input = {"food":'steak2', "beverage":'beer1'}
food_dim = ["cookies", "strawberries", "pasta", "steak", "burger", "oreos",
            "cookies1", "strawberries1", "pasta1", "steak1", "burger1", "oreos1",
            "cookies2", "strawberries2", "pasta2", "steak2", "burger2", "oreos2"]
bev_dim = ["milk","water","coffee","hot chocolate", "orange juice","beer",
           "milk1", "water1", "coffee1", "hot chocolate1", "orange juice1", "beer1",
           "milk2", "water2", "coffee2", "hot chocolate2", "orange juice2", "beer2",]
# food_dim = ["cookies", "strawberries", "pasta", "steak", "burger", "oreos"]
# bev_dim = ["milk","water","coffee","hot chocolate", "orange juice","beer"]
dimensions = {"food_dim":food_dim,"bev_dim":bev_dim}

# Define the initial input
input_dict = {'input':my_input, 'dimensions':dimensions}

# How many times to run the workflow + optimization loop
run_num = 5

# Or dynamically call till within a min_val
min_val = 0.397887
tolerance = 1.05

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
        gp_best.append(manageDB.get_optima('f', min_or_max='max')[0])
    manageDB.nuke()
    launchpad.defuse_wf(launchpad.get_fw_ids()[-1])

    # Run run_num iterations using a dummy optimizer (returns random)
    dummy_best = []
    wf = workflow_creator(input_dict, 'dummy')
    launchpad.add_wf(wf)
    for i in range(run_num):
        launch_rocket(launchpad)
        dummy_best.append(manageDB.get_optima('f', min_or_max='max')[0])
    manageDB.nuke()

    iterations = list(range(run_num))
    print("GP best:", gp_best[-1])
    print("Dummy best: ", dummy_best[-1])
    plt.plot(iterations,gp_best,'g', iterations, dummy_best,'r')
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
    gp_best = manageDB.get_optima('f', min_or_max='max')[0]
    gp_average = manageDB.get_avg('f')
    gp_total = manageDB.get_param('f')
    manageDB.nuke()

    # Run run_num iterations using a dummy optimizer (returns random)
    launchpad.defuse_wf(launchpad.get_fw_ids()[-1])
    wf = workflow_creator(input_dict, 'dummy')
    launchpad.add_wf(wf)
    rapidfire(launchpad, FWorker(), nlaunches=run_num, sleep_time=0)
    dummy_best = manageDB.get_optima('f', min_or_max='max')[0]
    dummy_average = manageDB.get_avg('f')
    dummy_total = manageDB.get_param('f')
    manageDB.nuke()

    # Compare the two optimizations graphically
    print('GP average: ', gp_average, '\n GP best:    ', gp_best)
    print('Dummy average: ', dummy_average, '\n Dummy best: ', dummy_best)
    iterations = list(range(run_num))
    plt.plot(iterations, gp_total, 'go', iterations, dummy_total, 'ro')
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
    gp_best.append(manageDB.get_optima('f', min_or_max='min')[0])

    while (gp_best[-1]>=tolerance*min_val):
        gp_iter = gp_iter+1
        launch_rocket(launchpad)
        gp_best.append(manageDB.get_optima('f', min_or_max='min')[0])

    manageDB.nuke()

    # Run some number of iterations until dummy iteration has converged
    launchpad.defuse_wf(launchpad.get_fw_ids()[-1])
    dummy_best = []
    dummy_iter = 0
    wf = workflow_creator(input_dict, 'dummy')
    launchpad.add_wf(wf)
    dummy_iter = dummy_iter + 1
    launch_rocket(launchpad)
    dummy_best.append(manageDB.get_optima('f', min_or_max='min')[0])

    while (dummy_best[-1] >= tolerance * min_val):
        dummy_iter = dummy_iter + 1
        launch_rocket(launchpad)
        dummy_best.append(manageDB.get_optima('f', min_or_max='min')[0])

    manageDB.nuke()

    print("GP iterations:", gp_iter)
    print("Dummy iterations:", dummy_iter)

    plt.plot(list(range(gp_iter)), gp_best, 'g', list(range(dummy_iter)), dummy_best, 'r')
    plt.show()

def testing_for_errors():
    """
    only for dev. bugfixing
    """
    wf = workflow_creator(input_dict,'skopt_gp')
    launchpad.add_wf(wf)
    rapidfire(launchpad, FWorker(), nlaunches=run_num, sleep_time=0)
    gp_max = manageDB.get_optima('f','min')
    launchpad.defuse_wf(launchpad.get_fw_ids()[-1])
    print (gp_max)

if __name__=="__main__":
    best_graph()