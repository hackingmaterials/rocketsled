from __future__ import unicode_literals, print_function, division

"""
A file for testing the workflow capabilities of OptTask.
Note that a local mongod instance in admin mode must be running for the tests to
pass by default.
WARNING: Tests reset the launchpad you specify. Specify a launchpad for testing 
you wouldn't mind resetting (e.g., mlab.com)
Modify tests_launchpad.yaml to define the db where you'd like to run the tests 
if you do not have access to admin mongod privledges on your local machine. 
"""
import os
import unittest
import yaml
import numpy as np
from fireworks import FWAction, Firework, Workflow, LaunchPad, ScriptTask
from fireworks.core.rocket_launcher import launch_rocket
from fireworks.core.firework import FireTaskBase
from fireworks.utilities.fw_utilities import explicit_serialize
from fw_tutorials.firetask.addition_task import AdditionTask
from rocketsled.opt import OptTask

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"

test_names = ['test_basic', 'test_custom_predictor', 'test_complex',
              'test_duplicates', 'test_get_z']

@explicit_serialize
class BasicTestTask(FireTaskBase):
    _fw_name = "BasicTestTask"

    def run_task(self, fw_spec):
        x = fw_spec['_x_opt']
        y = np.sum(x[:-1])        # sum all except the final string element
        return FWAction(update_spec={'_y_opt': y})

def wf_creator_basic(x, launchpad):
    """
    Testing a basic workflow with one Firework, and two FireTasks.
    """

    spec = {'_x_opt': x}
    dims = [(1, 10), (10.0, 20.0), ['blue', 'green', 'red', 'orange']]
    bt = BasicTestTask()
    ot = OptTask(wf_creator='rocketsled.tests.tests.wf_creator_basic',
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
    Testing a custom predictor which returns the same x vector for every guess,
    using same workflow as test_basic.
    """
    spec = {'_x_opt': x}
    dims = [(1, 10), (10.0, 20.0), ['blue', 'green', 'red', 'orange']]
    bt = BasicTestTask()
    ot = OptTask(wf_creator='rocketsled.tests.tests.wf_custom_predictor',
                 dimensions=dims,
                 predictor='rocketsled.tests.tests.custom_predictor',
                 lpad=launchpad,
                 wf_creator_args=[launchpad],
                 opt_label='test_custom_predictor')
    firework1 = Firework([bt, ot], spec=spec)
    return Workflow([firework1])

def wf_creator_complex(x, launchpad):
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

    spec = {'_x_opt': x}
    dims = [(1, 10), (10.0, 20.0), ['blue', 'green', 'red', 'orange']]
    fw0 = Firework(AdditionTask(), spec={"input_array": [1, 2]}, name='Parent')
    fw1 = Firework(AdditionTask(), spec={"input_array": [2, 3]}, name='Child A')
    fw2 = Firework(AdditionTask(), spec={"input_array": [3, 4]}, name='Child B')

    bt = BasicTestTask()
    ot = OptTask(wf_creator='rocketsled.tests.tests.wf_creator_complex',
                 dimensions=dims,
                 lpad=launchpad,
                 wf_creator_args=[launchpad],
                 opt_label='test_complex')
    fw3 = Firework([bt, ot], spec=spec, name="Optimization")

    fw4 = Firework(AdditionTask(), spec={"input_array": [5, 6]}, name='After 1')
    fw5 = Firework(ScriptTask.from_str('echo "ScriptTask: Finished complex '
                                       'workflow w/ optimization."'),
                   name='After 2')

    return Workflow([fw0, fw1, fw2, fw3, fw4, fw5],
                    {fw0: [fw1, fw2], fw1: [fw3], fw2: [fw3], fw3: [fw4], fw4:
                        [fw5], fw5: []})

def wf_creator_duplicates(x, launchpad):
    """
    Test workflow for duplicate checking with tolerances.
    """
    spec = {'_x_opt': x}
    dims = [(1, 10), (10.0, 20.0), ['blue', 'green', 'red', 'orange']]
    bt = BasicTestTask()
    ot = OptTask(wf_creator='rocketsled.tests.tests.wf_creator_duplicates',
                 dimensions=dims,
                 predictor='rocketsled.tests.tests.custom_predictor',
                 lpad=launchpad,
                 duplicate_check=True,
                 tolerances=[0, 1e-6, None],
                 wf_creator_args=[launchpad],
                 opt_label='test_duplicates')
    firework1 = Firework([bt, ot], spec=spec)
    return Workflow([firework1])

def wf_creator_get_z(x, launchpad):
    """
    Testing a basic workflow with one Firework, and two FireTasks with a get_z
    function. Also tests that duplicate checking is working with get_z.
    """
    spec = {'_x_opt': x}
    dims = [(1, 10), (10.0, 20.0), ['blue', 'green', 'red', 'orange']]
    bt = BasicTestTask()
    ot = OptTask(wf_creator='rocketsled.tests.tests.wf_creator_get_z',
                 dimensions=dims,
                 predictor='rocketsled.tests.tests.custom_predictor',
                 lpad=launchpad,
                 duplicate_check=True,
                 tolerances=[0, 1e-6, None],
                 get_z='rocketsled.tests.tests.get_z',
                 wf_creator_args=[launchpad],
                 opt_label='test_get_z')
    firework1 = Firework([bt, ot], spec=spec)
    return Workflow([firework1])

def custom_predictor(*args, **kwargs):
    return [3, 12.0, 'green']

def get_z(x):
    return [x[0] ** 2, x[1] ** 2]


class TestWorkflows(unittest.TestCase):
    def setUp(self):
        lp_filedir = os.path.dirname(os.path.realpath(__file__))
        lp_file = open(lp_filedir + '/tests_launchpad.yaml', 'r')
        lp_dict = yaml.load(lp_file)
        self.lp = LaunchPad.from_dict(lp_dict)
        self.db = self.lp.db

    def test_basic(self):
        self.lp.reset(password=None, require_password=False)
        self.lp.add_wf(wf_creator_basic([5, 11, 'blue'], self.lp))
        launch_rocket(self.lp)

        col = self.db.test_basic
        manager = col.find_one({'y': {'$exists': 0}})
        done = col.find_one({'y': {'$exists': 1, '$ne': 'reserved'}})
        reserved = col.find_one({'y': 'reserved'})

        self.assertEqual(col.find({}).count(), 3)
        self.assertEqual(manager['lock'], None)
        self.assertEqual(manager['queue'], [])
        self.assertEqual(done['x'], [5, 11, 'blue'])
        self.assertEqual(done['index'], 1)

    def test_custom_predictor(self):
        self.lp.reset(password=None, require_password=False)
        self.lp.add_wf(wf_custom_predictor([5, 11, 'blue'], self.lp))
        launch_rocket(self.lp)

        col = self.db.test_custom_predictor
        manager = col.find_one({'y': {'$exists': 0}})
        done = col.find_one({'y': {'$exists': 1, '$ne': 'reserved'}})
        reserved = col.find_one({'y': 'reserved'})

        self.assertEqual(col.find({}).count(), 3)
        self.assertEqual(manager['lock'], None)
        self.assertEqual(manager['queue'], [])
        self.assertEqual(done['x'], [5, 11, 'blue'])
        self.assertEqual(done['x_new'], [3, 12, 'green'])
        self.assertEqual(done['index'], 1)
        self.assertEqual(reserved['x'], [3, 12, 'green'])

    def test_complex(self):
        self.lp.reset(password=None, require_password=False)
        self.lp.add_wf(wf_creator_complex([5, 11, 'blue'], self.lp))
        for _ in range(10):
            launch_rocket(self.lp)

        col = self.db.test_complex
        loop1 = col.find({'index': 1})   # should return one doc, for first WF
        loop2 = col.find({'index': 2})   # should return one doc, for second WF
        reserved = col.find({'y': 'reserved'})
        self.assertEqual(col.find({}).count(), 4)
        self.assertEqual(reserved.count(), 1)
        self.assertEqual(loop1.count(), 1)
        self.assertEqual(loop2.count(), 1)

    def test_duplicates(self):
        self.lp.reset(password=None, require_password=False)
        self.lp.add_wf(wf_creator_duplicates([5, 11, 'blue'], self.lp))
        for _ in range(2):
            launch_rocket(self.lp)

        col = self.db.test_duplicates
        loop1 = col.find({'x': [5, 11, 'blue']})
        # should return one doc, for the first WF

        loop2 = col.find({'x': [3, 12, 'green']})
        # should return one doc, for the second WF

        reserved = col.find({'y': 'reserved'})
        self.assertEqual(col.find({}).count(), 4)
        self.assertEqual(reserved.count(), 1)
        self.assertEqual(loop1.count(), 1)
        self.assertEqual(loop2.count(), 1) # no duplicates are in the db

    def test_get_z(self):
        self.lp.reset(password=None, require_password=False)
        self.lp.add_wf(wf_creator_get_z([5, 11, 'blue'], self.lp))
        for _ in range(2):
            launch_rocket(self.lp)

        col = self.db.test_get_z
        loop1 = col.find_one({'index': 1})
        loop2 = col.find_one({'index': 2})

        self.assertEqual(col.find({}).count(), 4)
        self.assertEqual(loop1['x'], [5, 11, 'blue'])
        self.assertEqual(loop1['z'], [25.0, 121.0])
        self.assertEqual(loop2['x'], [3, 12.0, 'green'])
        self.assertEqual(loop2['z'], [9, 144.0])

    def tearDown(self):
        self.lp.reset(password=None, require_password=False)
        for tn in test_names:
            self.db.drop_collection(tn)

        if self.lp.host=='localhost'\
                and self.lp.port==27017 \
                and self.lp.name=='rsled_tests':
            try:
                self.lp.connection.drop_database('rsled_tests')
            except:
                pass

        # Remove straggler FW.json file, if it exists...
        fwjson_fp = os.path.dirname(os.path.realpath(__file__)) + '/FW.json'
        try:
            os.remove(fwjson_fp)
        except:
            pass

def suite():
    wf_test_suite = unittest.TestSuite()
    for tn in test_names:
        wf_test_suite.addTest(TestWorkflows(tn))
    return wf_test_suite


if __name__ == "__main__":
    unittest.main()