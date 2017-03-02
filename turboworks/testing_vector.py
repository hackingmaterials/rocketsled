from fireworks import Firework, LaunchPad, FWAction, FireTaskBase
from fireworks.core.rocket_launcher import launch_rocket
from fireworks.utilities.fw_utilities import explicit_serialize

from old.optimize_task import OptimizeTaskFromVector


@explicit_serialize
class CalculateTask(FireTaskBase):
    _fw_name = "CalculateTask"

    def run_task(self, fw_spec):
        A = fw_spec['input']['A']
        B = fw_spec['input']['B']
        C = fw_spec['input']['C']

        D_output = {'output': {'D': A * B / C}}
        return FWAction(update_spec=D_output)



def workflow_creator(inputs):


    dimensions = [(1,100), (1,100), (1,100)]
    spec = {'A':inputs[0], 'B':inputs[1], 'C':inputs[2], 'dimensions':dimensions}

    ft1 = CalculateTask()
    ft2 = OptimizeTaskFromVector(workflow_creator)
    firework = Firework([ft1, ft2], spec=spec)


if __name__ == "__main__":
    # mdb = ManageDB()
    # mdb.nuke()

    # set up the LaunchPad and reset it
    launchpad = LaunchPad()
    launchpad.reset('', require_password=False)

    # create the Firework consisting of a single task
    print type(OptimizeTaskFromVector)
    ft1 = OptimizeTaskFromVector(input = [1,2,3,4])
    fw1 = Firework(ft1)
    launchpad.add(fw1)
    launch_rocket(launchpad)