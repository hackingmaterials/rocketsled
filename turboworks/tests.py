"""
A file for testing the workflow capabilities of OptTask.
"""
import unittest
from examples.test_basic import run_workflows as run_basic
from examples.test_categorical import run_workflows as run_categorical
from examples.test_extras import run_workflows as run_extras

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"


class WorkflowsTestCase(unittest.TestCase):

    def test_basic(self):
        run_basic(test_case=True)

    def test_categorical(self):
        run_categorical(test_case=True)

    def test_extras(self):
        run_extras(test_case=True)

def suite():
    wf_test_suite = unittest.TestSuite()
    wf_test_suite.addTest(WorkflowsTestCase('test_basic'))
    wf_test_suite.addTest(WorkflowsTestCase('test_categorical'))
    wf_test_suite.addTest(WorkflowsTestCase('test_extras'))
    return wf_test_suite
