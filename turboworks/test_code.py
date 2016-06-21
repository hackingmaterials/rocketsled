from fireworks import Firework, FWorker, LaunchPad
from fireworks.core.rocket_launcher import rapidfire
from fireworks.core.rocket_launcher import launch_rocket
from optimize_task import OptimizeTask
from ABC_task import ABCtask

#Set up the launchpad
launchpad = LaunchPad()
launchpad.reset('', require_password=False)

#Sample data, 5 complete data points
A = [1.4, 5.4, 9.8, 12.2, 90.1]
B = [6.9, 4.4, 12.1, 39.2, 12.1]
C = [1.0, 6.9, 3.9, 51.0, 21.7]

#Assign FireTasks
firetask1 = ABCtask()
firetask2 = OptimizeTask()

#Execute FireWork
wf1 = [firetask1, firetask2]
firework1 = Firework(wf1, spec={"A_input":A, "B_input":B, "C_input":C})
launchpad.add_wf(firework1)
rapidfire(launchpad, FWorker())
