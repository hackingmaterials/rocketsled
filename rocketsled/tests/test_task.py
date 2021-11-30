"""
A file for testing the workflow capabilities of OptTask and MissionControl.

Note that a local mongod instance in admin mode must be running for the tests to
pass by default.

WARNING: Tests reset the launchpad you specify. Specify a launchpad for testing
you wouldn't mind resetting (e.g., mlab.com)
Modify tests_launchpad.yaml to define the db where you'd like to run the tests
if you do not have access to admin mongod privledges on your local machine.
"""
import os
import random
import unittest
import warnings

import numpy as np
import pymongo
from fireworks import Firework, FWAction, LaunchPad, ScriptTask, Workflow
from fireworks.core.firework import FireTaskBase
from fireworks.core.rocket_launcher import launch_rocket, rapidfire
from fireworks.scripts.rlaunch_run import launch_multiprocess
from fireworks.utilities.fw_utilities import explicit_serialize
from fw_tutorials.firetask.addition_task import AdditionTask
from ruamel.yaml import YAML

from rocketsled import MissionControl, OptTask
from rocketsled.utils import ExhaustedSpaceError

__author__ = "Alexander Dunn"
__email__ = "ardunn@lbl.gov"

lp_filedir = os.path.dirname(os.path.realpath(__file__))
with open(lp_filedir + "/tests_launchpad.yaml", "r") as lp_file:
    yaml = YAML()
    lp_dict = dict(yaml.load(lp_file))
    launchpad = LaunchPad.from_dict(lp_dict)
opt_label = "test"
db_info = {"launchpad": launchpad, "opt_label": opt_label}
test_db_name = launchpad.db
common_kwargs = {"predictor": "RandomForestRegressor", "acq": None}


@explicit_serialize
class BasicTestTask(FireTaskBase):
    _fw_name = "BasicTestTask"

    def run_task(self, fw_spec):
        x = fw_spec["_x"]
        y = np.sum(x[:-1])  # sum all except the final string element
        return FWAction(update_spec={"_y": y})


@explicit_serialize
class AccuracyTask(FireTaskBase):
    _fw_name = "AccuracyTask"

    def run_task(self, fw_spec):
        x = fw_spec["_x"]
        y = x[0] * x[1] / x[2]
        return FWAction(update_spec={"_y": y})


@explicit_serialize
class MultiTestTask(FireTaskBase):
    _fw_name = "MultiTestTask"

    def run_task(self, fw_spec):
        x = fw_spec["_x"]
        y = [np.sum(x[:-1]), np.prod(x[:-1])]
        return FWAction(update_spec={"_y": y})


def wf_creator_basic(x):
    """Testing a basic workflow with one Firework, and two FireTasks."""
    spec = {"_x": x}
    bt = BasicTestTask()
    ot = OptTask(**db_info)
    firework1 = Firework([bt, ot], spec=spec)
    return Workflow([firework1])


def wf_creator_complex(x):
    """
    Testing a custom workflow of five fireworks with complex dependencies, and
    optimization in the middle.
    This "complex" Workflow has the form:
                    fw0
                    / \
                  fw1 fw2
                   \  /
                   fw3 (optimization)
                    |
                   fw4
                    |
                   fw5
    """

    spec = {"_x": x}
    fw0 = Firework(AdditionTask(), spec={"input_array": [1, 2]}, name="Parent")
    fw1 = Firework(AdditionTask(), spec={"input_array": [2, 3]}, name="Child A")
    fw2 = Firework(AdditionTask(), spec={"input_array": [3, 4]}, name="Child B")
    bt = BasicTestTask()
    ot = OptTask(**db_info)
    fw3 = Firework([bt, ot], spec=spec, name="Optimization")
    fw4 = Firework(AdditionTask(), spec={"input_array": [5, 6]}, name="After 1")
    fw5 = Firework(
        ScriptTask.from_str(
            'echo "ScriptTask: Finished complex ' 'workflow w/ optimization."'
        ),
        name="After 2",
    )

    return Workflow(
        [fw0, fw1, fw2, fw3, fw4, fw5],
        {fw0: [fw1, fw2], fw1: [fw3], fw2: [fw3], fw3: [fw4], fw4: [fw5], fw5: []},
    )


def wf_creator_accuracy(x):
    """
    An expensive test ensuring the default predictor actually performs better
    than the average random case on the function defined in AccuracyTask.
    """
    spec = {"_x": x}
    at = AccuracyTask()
    ot = OptTask(**db_info)
    firework1 = Firework([at, ot], spec=spec)
    return Workflow([firework1])


def wf_creator_multiobjective(x):
    """
    Testing a multiobjective optimization.
    """

    spec = {"_x": x}
    mt = MultiTestTask()
    ot = OptTask(**db_info)
    firework1 = Firework([mt, ot], spec=spec)
    return Workflow([firework1])


def custom_predictor(*args, **kwargs):
    return [3, 12.0, "green"]


def get_z(x):
    return [x[0] ** 2, x[1] ** 2]


def wf_creator_batch(x):
    spec = {"_x": x}
    bt = BasicTestTask()
    ot = OptTask(**db_info)
    firework1 = Firework([bt, ot], spec=spec)
    return Workflow([firework1])


def custom_predictor_batch(XZ_explored, Y, x_dims, XZ_unexplored, batch_size=1):
    """
    Testin a custom predictor function for a batch optimization.

    Just uses a random sample of the unexplored space to return batch_size
    predictions.

    Args:
        XZ_explored:
        Y:
        x_dims:
        XZ_unexplored:
        batch_size:

    Returns:
        (list): ([3-dim array] * batch size) guesses.

    """
    x = random.sample(XZ_unexplored, k=batch_size)
    return x


class TestWorkflows(unittest.TestCase):
    def setUp(self):
        launchpad.reset(password=None, require_password=False)
        self.mc = MissionControl(**db_info)
        self.mc.reset(hard=True)
        self.db = launchpad.db
        self.dims_basic = [(1, 10), (10.0, 20.0), ["blue", "green", "red", "orange"]]
        self.dims_accuracy = [(1.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
        self.dims_parallel = [(1, 5), (1, 5), (1, 5)]

        self.c = getattr(self.db, opt_label)

    def test_basic(self):
        self.mc.configure(
            wf_creator=wf_creator_basic, dimensions=self.dims_basic, **common_kwargs
        )
        launchpad.add_wf(wf_creator_basic([5, 11, "blue"]))
        launch_rocket(launchpad)
        manager = self.c.find_one({"doctype": "manager"})
        done = self.c.find_one({"y": {"$exists": 1, "$ne": "reserved"}})
        reserved = {"y": "reserved"}

        self.assertEqual(self.c.count_documents({}), 4)
        self.assertEqual(manager["lock"], None)
        self.assertEqual(manager["queue"], [])
        self.assertEqual(done["x"], [5, 11, "blue"])
        self.assertEqual(done["index"], 1)
        self.assertEqual(self.c.count_documents(reserved), 1)

    def test_alternate_builtin_predictor(self):
        gp = "GaussianProcessRegressor"
        self.mc.configure(
            wf_creator=wf_creator_basic, dimensions=self.dims_basic, predictor=gp
        )
        config = self.c.find_one({"doctype": "config"})
        self.assertEqual(config["predictor"], gp)
        launchpad.add_wf(wf_creator_basic([5, 11, "blue"]))
        launch_rocket(launchpad)
        done = self.c.find_one({"y": {"$exists": 1, "$ne": "reserved"}})
        self.assertEqual(
            done["predictor"], f"{gp} with acquisition: Expected Improvement"
        )

    def test_custom_predictor(self):
        self.mc.configure(
            wf_creator=wf_creator_basic,
            dimensions=self.dims_basic,
            predictor=custom_predictor,
        )
        launchpad.add_wf(wf_creator_basic([5, 11, "blue"]))
        launch_rocket(launchpad)

        manager = self.c.find_one({"doctype": "manager"})
        done = self.c.find_one({"y": {"$exists": 1, "$ne": "reserved"}})
        reserved = self.c.find_one({"y": "reserved"})

        self.assertEqual(self.c.count_documents({}), 4)
        self.assertEqual(manager["lock"], None)
        self.assertEqual(manager["queue"], [])
        self.assertEqual(done["x"], [5, 11, "blue"])
        self.assertEqual(done["x_new"], [3, 12, "green"])
        self.assertEqual(done["index"], 1)
        self.assertEqual(reserved["x"], [3, 12, "green"])

    def test_complex(self):
        self.mc.configure(
            wf_creator=wf_creator_complex,
            dimensions=self.dims_basic,
            **common_kwargs,
        )
        launchpad.add_wf(wf_creator_complex([5, 11, "blue"]))
        for _ in range(10):
            launch_rocket(launchpad)

        self.assertEqual(self.c.count_documents({}), 5)
        self.assertEqual(self.c.count_documents({"y": "reserved"}), 1)
        # should return one doc, for first WF
        self.assertEqual(self.c.count_documents({"index": 1}), 1)  # loop 1
        # should return one doc, for second WF
        self.assertEqual(self.c.count_documents({"index": 2}), 1)  # loop 2

    def test_duplicates(self):
        self.mc.configure(
            wf_creator=wf_creator_basic,
            dimensions=self.dims_basic,
            duplicate_check=True,
            tolerances=[0, 1e-6, None],
            predictor=custom_predictor,
            acq=None,
        )
        launchpad.add_wf(wf_creator_basic([5, 11, "blue"]))
        for _ in range(2):
            launch_rocket(launchpad)

        self.assertEqual(self.c.count_documents({}), 5)
        self.assertEqual(self.c.count_documents({"y": "reserved"}), 1)
        # should return one doc, for the first WF
        self.assertEqual(self.c.count_documents({"x": [5, 11, "blue"]}), 1)
        # should return one doc, for the second WF
        # no duplicates are in the db
        self.assertEqual(self.c.count_documents({"x": [3, 12, "green"]}), 1)

    def test_get_z(self):
        self.mc.configure(
            wf_creator=wf_creator_basic,
            dimensions=self.dims_basic,
            predictor=custom_predictor,
            duplicate_check=True,
            tolerances=[0, 1e-6, None],
            get_z=get_z,
            acq=None,
        )

        launchpad.reset(password=None, require_password=False)
        launchpad.add_wf(wf_creator_basic([5, 11, "blue"]))
        for _ in range(2):
            launch_rocket(launchpad)

        loop1 = self.c.find_one({"index": 1})
        loop2 = self.c.find_one({"index": 2})

        self.assertEqual(self.c.count_documents({}), 5)
        self.assertEqual(loop1["x"], [5, 11, "blue"])
        self.assertEqual(loop1["z"], [25.0, 121.0])
        self.assertEqual(loop2["x"], [3, 12.0, "green"])
        self.assertEqual(loop2["z"], [9, 144.0])

    def test_accuracy(self):
        best = [None] * 10
        avg_random_best = 7.23002918931  # calculated with 1,000,000 calcs
        for n in range(10):
            self.mc.reset(hard=True)
            self.mc.configure(
                wf_creator=wf_creator_accuracy,
                dimensions=self.dims_accuracy,
                maximize=True,
                **common_kwargs,
            )
            launchpad.reset(password=None, require_password=False)
            launchpad.add_wf(wf_creator_accuracy([1, 10.1, 30.1]))
            for _ in range(20):
                launch_rocket(launchpad)
            # We want to maximize the function. The minimum is 0.5, and the
            # maximum is 10.
            for doc in self.c.find(
                {"y": {"$exists": 1, "$ne": "reserved"}},
                sort=[("y", pymongo.DESCENDING)],
                limit=1,
            ):
                best[n] = doc["y"]
        print(best)
        self.assertGreater(np.mean(best), avg_random_best)
        config = self.c.find_one({"doctype": "config"})
        self.assertEqual(config["predictor"], common_kwargs["predictor"])

    def test_multi(self):
        self.mc.configure(
            wf_creator=wf_creator_multiobjective,
            dimensions=self.dims_basic,
            predictor_kwargs={"random_state": 1},
            **common_kwargs,
        )
        launchpad.add_wf(wf_creator_multiobjective([5, 11, "blue"]))
        launch_rocket(launchpad)

        manager = self.c.find_one({"doctype": "manager"})
        done = self.c.find_one({"y": {"$exists": 1, "$ne": "reserved"}})
        # _ = self.c.find_one({"y": "reserved"})

        self.assertEqual(self.c.count_documents({}), 4)
        self.assertEqual(manager["lock"], None)
        self.assertEqual(manager["queue"], [])
        self.assertEqual(done["x"], [5, 11, "blue"])
        self.assertEqual(done["index"], 1)
        self.assertEqual(len(done["y"]), 2)

        # Loop 2, to make sure optimizations will keep running
        launch_rocket(launchpad)
        self.assertEqual(self.c.count_documents({}), 5)
        self.assertEqual(manager["lock"], None)
        self.assertEqual(manager["queue"], [])

    def test_batch(self):
        batch_size = 5
        self.mc.configure(
            wf_creator=wf_creator_batch,
            dimensions=self.dims_basic,
            predictor=custom_predictor_batch,
            predictor_kwargs={"batch_size": batch_size},
            batch_size=batch_size,
        )

        for i in range(batch_size):
            launchpad.add_wf(wf_creator_batch((1 + i, 10.0 + i, "green")))

        launch_rocket(launchpad)

        manager = self.c.find_one({"doctype": "manager"})
        done_q = {"y": {"$exists": 1, "$ne": "reserved"}}

        self.assertEqual(self.c.count_documents({}), 3)
        self.assertEqual(self.c.count_documents(done_q), 1)
        self.assertEqual(manager["lock"], None)
        self.assertEqual(manager["queue"], [])

        rapidfire(launchpad, nlaunches=4)

        manager = self.c.find_one({"doctype": "manager"})

        self.assertEqual(self.c.count_documents({}), 12)
        self.assertEqual(self.c.count_documents(done_q), 5)
        self.assertEqual(manager["lock"], None)
        self.assertEqual(manager["queue"], [])

        launch_rocket(launchpad)

        manager = self.c.find_one({"doctype": "manager"})

        self.assertEqual(self.c.count_documents({}), 12)
        self.assertEqual(self.c.count_documents(done_q), 6)
        self.assertEqual(manager["lock"], None)
        self.assertEqual(manager["queue"], [])

    def test_missioncontrol(self):
        self.mc.configure(
            wf_creator=wf_creator_basic, dimensions=self.dims_basic, **common_kwargs
        )
        launchpad.add_wf(wf_creator_basic([5, 11, "blue"]))
        rapidfire(launchpad, nlaunches=20, sleep_time=0)
        plt = self.mc.plot()
        savepath = os.path.join(lp_filedir, "missioncontrol_plot.png")
        plt.savefig(savepath)
        self.assertTrue(os.path.exists(savepath))

    def test_acqfuncs(self):
        self.mc.configure(
            wf_creator=wf_creator_basic,
            dimensions=self.dims_basic,
            predictor="GaussianProcessRegressor",
            acq=None,
        )
        launchpad.add_wf(wf_creator_basic([5, 11, "blue"]))
        rapidfire(launchpad, nlaunches=10, sleep_time=0)
        self.mc.reset(hard=False)
        self.mc.configure(
            wf_creator=wf_creator_basic,
            dimensions=self.dims_basic,
            predictor="GaussianProcessRegressor",
            acq="ei",
        )
        rapidfire(launchpad, nlaunches=2, sleep_time=0)
        self.mc.reset(hard=False)
        self.mc.configure(
            wf_creator=wf_creator_basic,
            dimensions=self.dims_basic,
            predictor="GaussianProcessRegressor",
            acq="pi",
        )
        rapidfire(launchpad, nlaunches=2, sleep_time=0)
        self.mc.reset(hard=False)
        self.mc.configure(
            wf_creator=wf_creator_basic,
            dimensions=self.dims_basic,
            predictor="GaussianProcessRegressor",
            acq="lcb",
        )
        rapidfire(launchpad, nlaunches=2, sleep_time=0)

    def test_parallel(self):
        n_procs = 10
        self.mc.configure(
            wf_creator=wf_creator_accuracy,
            dimensions=self.dims_parallel,
            opt_label="test_parallel",
            maximize=True,
            **common_kwargs,
        )

        for i in range(n_procs):
            # Assume the worst case, with n_procs forced duplicates
            launchpad.add_wf(wf_creator_accuracy([1, 5, 3]))

        launch_multiprocess(launchpad, None, "INFO", 12, n_procs, 0)
        try:
            for _ in range(10):
                launch_rocket(launchpad)
        except ExhaustedSpaceError:
            pass

        self.assertEqual(125, self.c.count_documents({"y": {"$exists": 1}}))

        all_x, all_y = self.mc.fetch_matrices(include_reserved=True)
        self.assertEqual(len(all_x), 125)
        self.assertEqual(len(all_x), 125)

        all_x, all_y = self.mc.fetch_matrices(include_reserved=False)
        self.assertGreaterEqual(len(all_x), 114)
        self.assertGreaterEqual(len(all_y), 114)

    def tearDown(self):
        try:
            launchpad.reset(password=None, require_password=False)
        except Exception:
            warnings.warn(
                "LaunchPad {} could not be reset! There may be "
                "fireworks from these tests remaining on the "
                "LaunchPad.".format(launchpad.to_dict())
            )
        try:
            self.db.drop_collection(opt_label)
        except BaseException:
            pass

        if (
            launchpad.host == "localhost"
            and launchpad.port == 27017
            and launchpad.name == test_db_name
        ):
            try:
                launchpad.connection.drop_database(test_db_name)
            except BaseException:
                pass


if __name__ == "__main__":
    unittest.main()
