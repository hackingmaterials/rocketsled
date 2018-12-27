"""
An example of running optimization workflows in parallel.

To ensure duplicate checking, optimizations themselves must be run sequentially
while the black box function evaluation can be run in parallel. To run
optimizations in parallel (and disable duplicate checking), change
enforce_sequential and duplicate_checking to False.

--------------------------------------------------------------------------
The following workflow is only one Firework (one job), for example purposes.
However, FireWorks and rocketsled are capable of handling more complex
workflows including multiple jobs and advanced dependencies. Please see the
Fireworks and rocketsled documentation pages for more information:

https://hackingmaterials.github.io/rocketsled/
https://materialsproject.github.io/fireworks/
"""

from fireworks import Workflow, Firework, LaunchPad
from fireworks.scripts.rlaunch_run import launch_multiprocess

from rocketsled import OptTask
from rocketsled.utils import random_guess
from rocketsled.examples.tasks import SumTask

lpad = LaunchPad(name='rsled')
dims = [(1, 5), (1, 5), (1, 5)]
Z_dim = dims


# a workflow creator function which takes x and returns a workflow based on x
def wf_creator(x):
    spec = {'_x': x, '_add_launchpad_and_fw_id': True}

    firework1 = Firework([SumTask(),
                          OptTask(
                              wf_creator='rocketsled.examples.parallel.wf_creator',
                              dimensions=Z_dim,
                              lpad=lpad,
                              duplicate_check=False,
                              opt_label="opt_parallel",
                              enforce_sequential=False)],
                         spec=spec)
    return Workflow([firework1])


# try a parallel implementation of rocketsled
def load_parallel_wfs(n_processes):
    for i in range(n_processes):
        lpad.add_wf(wf_creator(random_guess(dims)))


if __name__ == "__main__":
    lpad.reset(password=None, require_password=False)
    n_processes = 10
    n_runs = 13

    # Should throw an ExhaustedSpaceError when n_processes*n_runs > 125
    # (the total space size)
    load_parallel_wfs(n_processes)
    launch_multiprocess(lpad, None, 'INFO', n_runs, n_processes, 0)
