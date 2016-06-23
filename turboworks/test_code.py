import manage_DB
from workflow_creator import workflow_creator
from fireworks.core.rocket_launcher import rapidfire
from fireworks import FWorker, LaunchPad

#Set up the launchpad
launchpad = LaunchPad()
launchpad.reset('', require_password=False)

#Sample data, 5 complete data points (any size vector works here)
A = [1.4, 15.4, 39.8, 42.2, 90.1]
B = [6.9, 40.4, 32.1, 39.2, 12.1]
C = [91.0, 6.9, 63.9, 51.0, 21.7]
# A = [90.1]
# B = [12.1]
# C = [21.7]

#How many times we want to run the optimization iteration
opt_num = 5

#Create a workflow and run it
wf = workflow_creator(A,B,C)
launchpad.add_wf(wf)
rapidfire(launchpad, FWorker(), nlaunches=opt_num)


#Ask the DB to do stuff if we want to
manage_DB.countit()
# manage_DB.nukeit()
