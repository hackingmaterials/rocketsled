from fireworks import Firework, LaunchPad, FWAction, FireTaskBase, Workflow
from turboworks.optimize_task import OptimizeTask
from turboworks.manage_db import ManageDB
from turboworks.gp_opt import gp_minimize
from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.rocket_launcher import launch_rocket
from turboworks.reference import ref_dict


from pprint import pprint

@explicit_serialize
class CalculateTask(FireTaskBase):
    _fw_name = "CalculateTask"

    def run_task(self, fw_spec):
        A = fw_spec['Structure']['A']
        B = fw_spec['Structure']['B']
        C = fw_spec['Structure']['C']

        D_output = {'output': {'D': A * B / C}}
        # D_output = {'output': {'D':A*C}}
        # Modify changes in spec
        return FWAction(update_spec=D_output)

@explicit_serialize
class SkoptimizeTask(OptimizeTask):
    _fw_name = "SkoptimizeTask"

    def run_task(self, fw_spec):

        # self.store(fw_spec)
        # X = self.gather_single('input')
        # y = self.gather_single('output', type='list')
        # dim = self.gather_single('dim', type='dim')

        print(self.auto_extract(inputs=['Structure.A', 'e_above_hull', 'types.new.s']))

        # print(X)

        # update = self.deconsolidate(features = ['A', 'B', 'C'], matrix = autod)
        #todo: have function automatically make updated dictionary based on
        #todo: structure of fw_spec input




if __name__ == "__main__":

    # mdb = ManageDB()
    # mdb.nuke()

    # set up the LaunchPad and reset it
    launchpad = LaunchPad()
    launchpad.reset('', require_password=False)

    fw_spec = ref_dict


    # create the Firework consisting of a single task
    firetask1 = CalculateTask()
    firetask2 = SkoptimizeTask()

    # firework = Firework([firetask1, firetask2], spec=fw_spec)
    firework = Firework([firetask1, firetask2],  spec=fw_spec)
    # store workflow and launch it locally
    launchpad.add_wf(firework)
    launch_rocket(launchpad)
