import manage_DB
from ABC_workflow_creator import workflow_creator
from fireworks.core.rocket_launcher import rapidfire
from fireworks import FWorker, LaunchPad
from fireworks.core.rocket_launcher import launch_rocket
from manage_DB import count_it, nuke_it, query_it

#Set up the launchpad
launchpad = LaunchPad()
launchpad.reset('', require_password=False)

#Sample data, 5 complete data points (any size vector works here)
A = 44.3
B = 42.8
C = 12.3

#How many times to automatically run the optimization iteration
run_num = 3

#Create a workflow and run it
wf = workflow_creator({'A_input':A,'B_input':B,'C_input':C})
launchpad.add_wf(wf)
rapidfire(launchpad, FWorker(), nlaunches=run_num)
# launch_rocket(launchpad)

#Ask the DB to do stuff if we want to
count_it()
query_it()
nuke_it()

