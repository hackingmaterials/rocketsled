from fireworks import Firework, LaunchPad, FWAction, FireTaskBase, Workflow
from turboworks.optimize_task import OptimizeTask
from turboworks.manage_db import ManageDB
from turboworks.gp_opt import gp_minimize
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.rocket_launcher import launch_rocket


@explicit_serialize
class CalculateTask(FireTaskBase):
    _fw_name = "CalculateTask"

    def run_task(self, fw_spec):
        A = fw_spec['input']['A']
        B = fw_spec['input']['B']
        C = fw_spec['input']['C']

        D_output = {'output': {'D': A * B / C}}

        # Modify changes in spec
        return FWAction(update_spec=D_output)

@explicit_serialize
class SkoptimizeTask(OptimizeTask):
    _fw_name = "SkoptimizeTask"

    def run_task(self, fw_spec):
        self.store(fw_spec)
        X = self.to_list('input')
        y = self.to_list('output', type='list')
        dim = self.to_list('dim', type='dim')

        x = gp_minimize(X,y,dim)

        update = self.to_vars()



if __name__ == "__main__":

    # mdb = ManageDB()
    # mdb.nuke()

    # set up the LaunchPad and reset it
    launchpad = LaunchPad()
    launchpad.reset('', require_password=False)

    fw_spec = {'input': {'A': 1.0, 'B': 2.0, 'C': 3.0},
               'dim': {'A':(1,100), 'B':(1,100), 'C':(12,30)}}
    param_list = {'data': ['X', 'y'], 'hyperparameters': []}

    # create the Firework consisting of a single task
    firetask1 = CalculateTask()
    firetask2 = SkoptimizeTask()

    # firework = Firework([firetask1, firetask2], spec=fw_spec)
    firework = Firework([firetask1, firetask2],  spec=fw_spec)
    # store workflow and launch it locally
    launchpad.add_wf(firework)
    launch_rocket(launchpad)
