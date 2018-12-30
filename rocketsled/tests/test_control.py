"""
Testing MissionControl setup and management tool.
"""
import os
import unittest

from fireworks import LaunchPad

from rocketsled import MissionControl


class TestMissionControl(unittest.TestCase):

    def setUp(self):
        lp_filedir = os.path.dirname(os.path.realpath(__file__))
        lp_file = os.path.join(lp_filedir, '/tests_launchpad.yaml')
        self.launchpad = LaunchPad.from_file(lp_file)
        self.opt_label = "opt_label"
        self.mc = MissionControl(launchpad=self.launchpad,
                                 opt_label=self.opt_label)

    def test_configure(self):
        # test with multiple kinds of func types (str, obj) passed in
        pass

    def test_reset(self):
        pass

    def test_summary(self):
        pass

    def test_plotting(self):
        pass
