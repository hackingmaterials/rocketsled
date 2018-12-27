"""
Testing utility functions in rocketsled
"""
import os
import unittest

import numpy as np

from rocketsled.utils import Dtypes, random_guess, pareto, \
    convert_value_to_native, latex_float, deserialize, split_xz


class TestUtilities(unittest.TestCase):
    def test_Dtypes(self):
        dt = Dtypes()
        self.assertTrue(int in dt.ints)
        self.assertTrue(np.int in dt.ints)
        self.assertTrue(float in dt.floats)
        self.assertTrue(np.float in dt.floats)
        self.assertTrue(str in dt.all)
        self.assertTrue(str in dt.others)
        self.assertTrue(bool in dt.all)
        self.assertTrue(bool in dt.bool)
        self.assertTrue(np.bool in dt.bool)

    def test_random_guess(self):
        d1 = tuple([1, 5])
        d2 = ["red", "green", "blue"]
        dims = [d1, d2]
        rg = random_guess(dims)
        self.assertTrue(rg[0] in [1, 2, 3, 4, 5])
        self.assertTrue(rg[1] in d2)

    def test_pareto(self):
        test_arr = np.asarray([[5, 5], [2, 2], [1, 4], [3, 2]])
        mins = test_arr[pareto(test_arr, maximize=False)]
        maxes = test_arr[pareto(test_arr, maximize=True)]
        self.assertTrue([2, 2] in mins)
        self.assertTrue([1, 4] in mins)
        self.assertFalse(([3, 2] in maxes))
        self.assertTrue([5, 5] in maxes)

    def test_convesion(self):
        a = np.int(5)
        b = np.float(1.4)
        c = np.str("somestr")
        a_native = convert_value_to_native(a)
        b_native = convert_value_to_native(b)
        c_native = convert_value_to_native(c)
        self.assertTrue(isinstance(a_native, int))
        self.assertTrue(isinstance(b_native, float))
        self.assertTrue(isinstance(c_native, str))

    def test_latex_float(self):
        f1 = 3.494388373744
        f2 = 3.223421e-16
        self.assertTrue(latex_float(f1), "3.49")
        self.assertTrue(latex_float(f2), "3.22 \times 10^{-16}")

    def test_deserialize(self):
        cwd = os.path.dirname(os.path.realpath(__file__))
        funcstr = cwd + "/deserialize_func.obj_func"
        f = deserialize(funcstr)
        self.assertEqual(f([1, 2, 3]), 6)
        self.assertAlmostEqual(f([1.0, 2.0, 3.0]), 6.0)

    def test_split_xz(self):
        x_dims = [(1, 10), (1, 10), (1, 10)]
        x = [1, 2, 3]
        z = ["red", "monkey"]
        xz = x + z
        x_split, z_split = split_xz(xz, x_dims)
        self.assertListEqual(x, x_split)
        self.assertListEqual(z, z_split)







