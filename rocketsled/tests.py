from __future__ import unicode_literals, print_function, division

"""
A file for testing the workflow capabilities of OptTask.
"""
import unittest
from rs_examples.test_basic import run_workflows as run_basic
from rs_examples.test_categorical import run_workflows as run_categorical
from rs_examples.test_extras import run_workflows as run_extras
from rs_examples.test_batch import run_workflows as run_batch

__author__ = "Alexander Dunn"
__version__ = "0.1"
__email__ = "ardunn@lbl.gov"


class TestWorkflows(unittest.TestCase):
    """
    Make sure basic optimization loops execute properly.
    """

    def test_basic(self):
        run_basic(test_case=True)

    def test_categorical(self):
        run_categorical(test_case=True)

    def test_extras(self):
        run_extras(test_case=True)

    def test_batch(self):
        run_batch(test_case=True)

def suite():
    wf_test_suite = unittest.TestSuite()
    wf_test_suite.addTest(TestWorkflows('test_basic'))
    wf_test_suite.addTest(TestWorkflows('test_categorical'))
    wf_test_suite.addTest(TestWorkflows('test_extras'))
    wf_test_suite.addTest(TestWorkflows('test_batch'))
    return wf_test_suite
