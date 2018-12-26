"""
Testing utility functions in rocketsled
"""
import os
import unittest

from fireworks import LaunchPad
from ruamel.yaml import YAML

from rocketsled import auto_setup


DIMS = [(1, 5), (2, 6)]

def testfun(x):
    """
    A test function returning the sum of a list.

    Args:
        x (list): Input

    Returns:
        Sum of x

    """
    return sum(x)


class TestAutoSleds(unittest.TestCase):
    def setUp(self):
        self.cwd = os.path.dirname(os.path.realpath(__file__))
        self.autosled_dir = os.path.join(self.cwd, "../auto_sleds")
        with open(self.cwd + '/tests_launchpad.yaml', 'r') as lp_file:
            yaml = YAML()
            lp_dict = dict(yaml.load(lp_file))
            self.lp = LaunchPad.from_dict(lp_dict)

    def test_basic(self):
        auto_setup(testfun, DIMS, wfname="TEST_AUTO_SLED")
        sled_file = os.path.join(self.autosled_dir, "TEST_AUTO_SLED.py")
        self.assertTrue(os.path.exists(sled_file))

    def test_with_lpad(self):
        auto_setup(testfun, DIMS, wfname="TEST_WITH_LPAD")
        sled_file = os.path.join(self.autosled_dir, "TEST_WITH_LPAD.py")
        self.assertTrue(os.path.exists(sled_file))

    def tearDown(self):
        for file in ["TEST_AUTO_SLED.py", "TEST_WITH_LPAD.py"]:
            pth = os.path.join(self.autosled_dir, file)
            try:
                os.remove(pth)
            except FileNotFoundError:
                continue









