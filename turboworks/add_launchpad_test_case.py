from fireworks.utilities.fw_utilities import explicit_serialize
from fireworks.core.firework import FireTaskBase
from fireworks import Firework, Workflow, LaunchPad

@explicit_serialize
class AddLaunchpadTask(FireTaskBase):
    _fw_name = "AddLaunchpadTask"

    def run_task(self, fw_spec):

        host, port, name = [getattr(self.launchpad, k) for k in ('host', 'port', 'name')]
        print(host, port, name)


if __name__=="__main__":


    launchpad = LaunchPad()
    launchpad.reset(password=None, require_password=False)
    fw = Firework([AddLaunchpadTask()], spec={'_add_launchpad_and_fw_id':True})
    launchpad.add_wf(fw)
    launchpad.add_wf(fw)