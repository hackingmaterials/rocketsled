

from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from fireworks.core.rocket_launcher import launch_rocket
from fireworks import Firework, LaunchPad, Workflow

@explicit_serialize
class PrintTask(FireTaskBase):
    _fw_name = "PrintTask"

    required_params = ['arg']

    def run_task(self, fw_spec):
        print "running PrintTask" + self['arg']

if __name__=="__main__":
    fw1 = Firework([PrintTask(arg='fw1task1'), PrintTask(arg='fw1task2')])
    fw2 = Firework([PrintTask(arg='fw2task1'), PrintTask(arg='fw2task2')])
    launchpad = LaunchPad()
    launchpad.reset('', require_password=False)
    launchpad.add_wf(fw1)
    launchpad.add_wf(fw2)


