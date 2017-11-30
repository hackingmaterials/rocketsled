from __future__ import unicode_literals, print_function, division

"""
A file for testing the workflow capabilities of OptTask.
"""
import unittest
import numpy as np
from pymongo import MongoClient
from fireworks import FWAction, Firework, Workflow, LaunchPad
from fireworks.core.rocket_launcher import launch_rocket
from fireworks.core.firework import FireTaskBase
from fireworks.utilities.fw_utilities import explicit_serialize
from rocketsled.optimize import OptTask

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"


# todo: test for complex workflow
# todo: test for parallel duplicates
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

    def tearDown(self):
        self.db.drop_database('rstest')


def suite():
    wf_test_suite = unittest.TestSuite()
    wf_test_suite.addTest(TestWorkflows('test_basic'))
    wf_test_suite.addTest(TestWorkflows('test_custom_predictor'))
    return wf_test_suite

