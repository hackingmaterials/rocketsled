from __future__ import unicode_literals, print_function, division

"""
A file for testing the workflow capabilities of OptTask.
"""
import unittest
import numpy as np
from pymongo import MongoClient
from fireworks import FWAction, Firework, Workflow, LaunchPad, ScriptTask
from fireworks.core.rocket_launcher import launch_rocket, rapidfire
from fireworks.core.firework import FireTaskBase
from fireworks.utilities.fw_utilities import explicit_serialize
from fw_tutorials.dynamic_wf.fibadd_task import FibonacciAdderTask
from fw_tutorials.firetask.addition_task import AdditionTask
from rocketsled.optimize import OptTask

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"


# todo: test for parallel duplicates
# todo: test for float/categorical/dtypes
# todo: test for get_z issues
# todo: test for less important params

@explicit_serialize
class BasicTestTask(FireTaskBase):
    _fw_name = "BasicTestTask"

    def run_task(self, fw_spec):
        x = fw_spec['_x_opt']
        y = np.sum(x)
        return FWAction(update_spec={'_y_opt': y})

def wf_creator_basic(x, launchpad):
    """
    Testing a basic workflow with one Firework, and two FireTasks.
    """

    spec = {'_x_opt': x}
    dims = [(1, 10), (10, 20), (20, 30)]
    bt = BasicTestTask()
    ot = OptTask(wf_creator='rocketsled.tests.wf_creator_basic',
                 dimensions=dims,
                 predictor='RandomForestRegressor',
                 predictor_kwargs={'random_state': 1},
                 lpad=launchpad,
                 wf_creator_args=[launchpad],
                 opt_label='test_basic')
    firework1 = Firework([bt, ot], spec=spec)
    return Workflow([firework1])

def wf_custom_predictor(x, launchpad):
    """
    Testing a custom predictor which returns the same x vector for every guess, using same workflow as test_basic.
    """
    spec = {'_x_opt': x}
    dims = [(1, 10), (10, 20), (20, 30)]
    bt = BasicTestTask()
    ot = OptTask(wf_creator='rocketsled.tests.wf_custom_predictor',
                 dimensions=dims,
                 predictor='rocketsled.tests.custom_predictor',
                 lpad=launchpad,
                 wf_creator_args=[launchpad],
                 opt_label='test_custom_predictor')
    firework1 = Firework([bt, ot], spec=spec)
    return Workflow([firework1])

def wf_creator_complex(x, launchpad):
    """
    Testing a custom workflow of five fireworks with complex dependencies, and optimization in the middle.
    """

    spec = {'_x_opt': x}
    dims = [(1, 10), (10, 20), (20, 30)]

    fw0 = Firework(AdditionTask(), spec={"input_array": [1, 2]}, name='Parent')
    fw1 = Firework(AdditionTask(), spec={"input_array": [2, 3]}, name='Child A')
    fw2 = Firework(AdditionTask(), spec={"input_array": [3, 4]}, name='Child B')

    bt = BasicTestTask()
    ot = OptTask(wf_creator='rocketsled.tests.wf_creator_complex',
                 dimensions=dims,
                 lpad=launchpad,
                 predictor='rocketsled.tests.custom_predictor',
                 wf_creator_args=[launchpad],
                 duplicate_check=True,
                 opt_label='test_complex')
    fw3 = Firework([bt, ot], spec=spec, name="Optimization")

    fw4 = Firework(AdditionTask(), spec={"input_array": [5, 6]}, name='After 1')
    fw5 = Firework(ScriptTask.from_str('echo "ScriptTask: Finished complex workflow w/ optimization."'), name='After 2')

    return Workflow([fw0, fw1, fw2, fw3, fw4, fw5],
                    {fw0: [fw1, fw2], fw1: [fw3], fw2: [fw3], fw3: [fw4], fw4: [fw5], fw5: []})

def custom_predictor(*args, **kwargs):
    return [3, 12, 25]


class TestWorkflows(unittest.TestCase):
    def setUp(self):
        self.lp = LaunchPad(name='rstest')
        self.db = MongoClient(self.lp.host, self.lp.port)

    def test_basic(self):
        self.lp.reset(password=None, require_password=False)
        self.lp.add_wf(wf_creator_basic([5, 11, 25], self.lp))
        launch_rocket(self.lp)

        col = self.db.rstest.test_basic
        manager = col.find_one({'y': {'$exists': 0}})
        done = col.find_one({'y': {'$exists': 1, '$ne': 'reserved'}})
        reserved = col.find_one({'y': 'reserved'})

        self.assertEqual(col.find({}).count(), 3)
        self.assertEqual(manager['lock'], None)
        self.assertEqual(manager['queue'], [])
        self.assertEqual(done['x'], [5, 11, 25])
        self.assertEqual(done['index'], 1)

    def test_custom_predictor(self):
        self.lp.reset(password=None, require_password=False)
        self.lp.add_wf(wf_custom_predictor([5, 11, 25], self.lp))
        launch_rocket(self.lp)

        col = self.db.rstest.test_custom_predictor
        manager = col.find_one({'y': {'$exists': 0}})
        done = col.find_one({'y': {'$exists': 1, '$ne': 'reserved'}})
        reserved = col.find_one({'y': 'reserved'})

        self.assertEqual(col.find({}).count(), 3)
        self.assertEqual(manager['lock'], None)
        self.assertEqual(manager['queue'], [])
        self.assertEqual(done['x'], [5, 11, 25])
        self.assertEqual(done['x_new'], [3, 12, 25])
        self.assertEqual(done['index'], 1)
        self.assertEqual(reserved['x'], [3, 12, 25])

    def test_complex(self):
        self.lp.reset(password=None, require_password=False)
        self.lp.add_wf(wf_creator_complex([5, 11, 25], self.lp))
        for _ in range(10):
            launch_rocket(self.lp)

        col = self.db.rstest.test_complex
        manager = col.find_one({'y': {'$exists': 0}})
        loop1 = col.find({'x': [5, 11, 25]})   # should return one doc, for the first WF
        loop2 = col.find({'x': [3, 12, 25]})   # should return one doc, for the second WF
        reserved = col.find({'y': 'reserved'})
        self.assertEqual(col.find({}).count(), 4)
        self.assertEqual(reserved.count(), 1)
        self.assertEqual(loop1.count(), 1)
        self.assertEqual(loop2.count(), 1)
        self.assertEqual(manager['lock'], None)
        self.assertEqual(manager['queue'], [])

    def tearDown(self):
        self.db.drop_database('rstest')


def suite():
    wf_test_suite = unittest.TestSuite()
    wf_test_suite.addTest(TestWorkflows('test_basic'))
    wf_test_suite.addTest(TestWorkflows('test_custom_predictor'))
    wf_test_suite.addTest(TestWorkflows('test_complex'))

    return wf_test_suite

