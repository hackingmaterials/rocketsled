from fireworks import Firework, LaunchPad, FWAction, FireTaskBase
from turboworks.optimize_task import OptimizeTask
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.rocket_launcher import launch_rocket

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

iris = load_iris()


@explicit_serialize
class CalculateTask(FireTaskBase):
    _fw_name = "CalculateTask"



    def run_task(self, fw_spec):


        A_input = fw_spec['input']['A']
        B_input = fw_spec['input']['B']
        C_input = fw_spec['input']['C']

        D_output = {'output': {'D': float(A_input * B_input / C_input)}}

        # Modify changes in spec
        return FWAction(update_spec=D_output)




if __name__ == "__main__":
    # set up the LaunchPad and reset it
    launchpad = LaunchPad()
    launchpad.reset('', require_password=False)

    fw_spec = {'input':{'A':1, 'B':2, 'C':3}}
    param_list = {'data':['X', 'y'], 'hyperparameters':[]}

    # create the Firework consisting of a single task
    firetask1 = CalculateTask()
    firetask2 = OptimizeTask()
    firework = Firework([firetask1, firetask2])

    # store workflow and launch it locally
    launchpad.add_wf(firework)
    launch_rocket(launchpad)
