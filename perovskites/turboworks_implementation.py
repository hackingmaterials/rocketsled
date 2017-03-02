from fireworks import Firework, LaunchPad, FWAction, FireTaskBase
from fireworks.core.rocket_launcher import launch_rocket
from fireworks.utilities.fw_utilities import explicit_serialize

from old.optimize_task import OptimizeTask
from turboworks.dummy_opt import dummy_minimize
from turboworks.reference import ref_dict


@explicit_serialize
class CalculateTask(FireTaskBase):
    _fw_name = "CalculateTask"

    def run_task(self, fw_spec):
        A = fw_spec['Structure']['A']
        B = fw_spec['Structure']['B']
        C = fw_spec['Structure']['C']

        D_output = {'energy': {'good_estimate': A * B / C}}
        # D_output = {'output': {'D':A*C}}
        # Modify changes in spec
        return FWAction(update_spec=D_output)

@explicit_serialize
class SkoptimizeTask(OptimizeTask):
    _fw_name = "SkoptimizeTask"

    def run_task(self, fw_spec):

        # Store new data
        self.store(fw_spec)

        # Extract the data we want from the database
        features = ['Structure']
        output = ['energy.good_estimate']

        X = self.auto_extract(features, label='inputs')
        y = self.auto_extract(output, label='outputs')

        # Run a machine learning algorithm on the data
        dimensions = [(0, 100), (0,100), (0,100)]
        y_new = dummy_minimize(dimensions)

        # y_new = gp_minimize(X,y,dimensions)

        # Update our workflow spec with the new data
        self.auto_update(y_new)

        # Return a workflow
        new_fw = Firework([CalculateTask(), SkoptimizeTask()], spec=self.tw_spec)
        return FWAction(additions=new_fw)





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
    firework = Firework([CalculateTask(), SkoptimizeTask()],  spec=fw_spec)
    # store workflow and launch it locally
    launchpad.add_wf(firework)


    # Repeatedly execute the optimization loop
    for i in range(1):
        launch_rocket(launchpad)
