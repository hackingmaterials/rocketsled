from fireworks import Firework, LaunchPad, FWAction, FireTaskBase, Workflow
from fireworks.core.rocket_launcher import launch_rocket
from fireworks.utilities.fw_utilities import explicit_serialize

from old.optimize_task import OptimizeTask, AutoOptimizeTask
from old.reference import ref_dict, ref_dict2
from turboworks.db import DB
from turboworks.dummy import dummy_minimize


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
        return FWAction(update_spec=D_output, stored_data=D_output)



@explicit_serialize
class NothingTask(FireTaskBase):
    _fw_name = "NothingTask"

    def run_task(self, fw_spec):
        print "final task result:", fw_spec['energy']['good_estimate']



@explicit_serialize
class SkoptimizeTask(OptimizeTask):
    _fw_name = "SkoptimizeTask"

    def run_task(self, fw_spec):

        # Store new data
        self.store(fw_spec)

        # Extract the data we want from the database
        features = ['Structure']
        # output = ['energy.good_estimate']

        X = self.auto_extract(features, label='inputs')
        # y = self.auto_extract(output, label='outputs')

        print X

        # Run a machine learning algorithm on the data
        dimensions = [(0, 100), (0,100), (0,100)]
        y_new = dummy_minimize(dimensions)

        # y_new = gp_minimize(X,y,dimensions)

        # Update our workflow spec with the new data
        self.auto_update(y_new)

        # Return a workflow
        fw = Firework([CalculateTask(), SkoptimizeTask()], spec=self.tw_spec)
        return FWAction(additions=fw)


if __name__ == "__main__":

    mdb = DB()
    mdb.clean()

    # set up the LaunchPad and reset it
    launchpad = LaunchPad()
    launchpad.reset('', require_password=False)

    # create the Firework consisting of a single task
    firetask1 = CalculateTask()
    firetask2 = NothingTask()

    # firework = Firework([firetask1, firetask2], spec=fw_spec)
    firework1 = Firework([CalculateTask(), NothingTask()], spec=ref_dict, name='Firework1')
    firework2 = Firework([CalculateTask(), NothingTask()], spec=ref_dict2, name='Firework2')
    wf = Workflow([firework1, firework2])

    firework3 = Firework([AutoOptimizeTask(workflow=wf, inputs = ['Firework1/Structure/A', 'Firework2/Structure/B'],
                                           outputs=['Firework2/energy/good_estimate'], dimensions=[])])


    # firework4 = Firework([SkoptimizeTask()], spec=ref_dict)
    # firework5 = Firework([SkoptimizeTask()], spec=ref_dict2)
    # wf = Workflow([firework1, firework2])
    #
    launchpad.add_wf(firework3)

    # Repeatedly execute the optimization loop
    for i in range(3):
        launch_rocket(launchpad)
