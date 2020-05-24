"""
Testing utility functions in rocketsled
"""
import os
import unittest

import numpy as np

from rocketsled.utils import (
    check_dims,
    convert_native,
    convert_value_to_native,
    deserialize,
    dtypes,
    get_default_opttask_kwargs,
    get_len,
    is_discrete,
    is_duplicate_by_tolerance,
    latex_float,
    pareto,
    random_guess,
    serialize,
    split_xz,
)


class TestUtilities(unittest.TestCase):
    def test_Dtypes(self):
        dt = dtypes
        self.assertTrue(int in dt.ints)
        self.assertTrue(np.int in dt.ints)
        self.assertTrue(float in dt.floats)
        self.assertTrue(np.float in dt.floats)
        self.assertTrue(type("somestr") in dt.all)
        self.assertTrue(type("somestr") in dt.others)
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

    def test_convert_value_to_native(self):
        a = np.int(5)
        b = np.float(1.4)
        c = np.str("somestr")
        a_native = convert_value_to_native(a)
        b_native = convert_value_to_native(b)
        c_native = convert_value_to_native(c)
        self.assertTrue(isinstance(a_native, int))
        self.assertTrue(isinstance(b_native, float))
        self.assertTrue(isinstance(c_native, str))

    def test_convert_native(self):
        a = [np.int(10), np.float(12.2), np.str("a str"), 12.3, 100, "ok"]
        native = convert_native(a)
        self.assertListEqual(
            [type(i) for i in native], [int, float, str, float, int, str]
        )

    def test_latex_float(self):
        f1 = 3.494388373744
        f2 = 3.223421e-16
        self.assertTrue(latex_float(f1), "3.49")
        self.assertTrue(latex_float(f2), "3.22 \times 10^{-16}")

    def test_serialize(self):
        fstr = "rocketsled.tests.deserialize_func.obj_func"
        from rocketsled.tests.deserialize_func import obj_func

        self.assertEqual(serialize(obj_func), fstr)
        self.assertEqual(serialize(obj_func), fstr)

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

    def test_get_default_opttask_kwargs(self):
        kwargs = get_default_opttask_kwargs()
        self.assertTrue(isinstance(kwargs, dict))

    def test_check_dims(self):
        good_dims = [(1, 50), (100.0, 200.0), ["orange", "blue"]]
        good_dims_2 = [
            [1.45, 1.43, 1.78, 1.98],
            ["orange", "blue"],
            [1, 12, 1400, 1975],
        ]
        bad_dims_1 = {"dim1": 12, "dim2": (100, 200)}
        bad_dims_2 = [{1.5: 200}, ["red", "green", "blue"]]
        bad_dims_3 = [("red", 12, 15), ["red", "greeen"]]
        bad_dims_4 = [(1.5, 200), (1.4, 1.7, 2.9)]
        self.assertListEqual(
            check_dims(good_dims), ["int_range", "float_range", "categorical 2"]
        )
        self.assertListEqual(
            check_dims(good_dims_2), ["float_set", "categorical 2", "int_set"]
        )
        with self.assertRaises(TypeError):
            check_dims(bad_dims_1)
        with self.assertRaises(TypeError):
            check_dims(bad_dims_2)
        with self.assertRaises(TypeError):
            check_dims(bad_dims_3)
        with self.assertRaises(TypeError):
            check_dims(bad_dims_4)

    def test_is_discrete(self):
        dims1 = [(1.0, 200.0), ["red", "green"], (1, 10)]
        dims2 = [(1, 10), (2, 20), [1, 2, 3, 4, 5], ["orange", "red"]]
        dims3 = [(1.00, 200.0), [1.5, 1.8, 1.9]]
        self.assertFalse(is_discrete(dims1, "all"))
        self.assertTrue(is_discrete(dims1, "any"))
        self.assertTrue(is_discrete(dims2, "all"))
        self.assertTrue(is_discrete(dims2, "any"))
        self.assertFalse(is_discrete(dims3, "all"))
        self.assertFalse(is_discrete(dims3, "any"))

    def test_tolerance_check(self):
        tolerances = [1e-6, None, 2]
        all_x_explored = [[1.45, "red", 201], [1.48, "green", 209]]
        x_dup = [1.4500001, "red", 203]
        x_clear = [1.45, "green", 220]
        self.assertFalse(
            is_duplicate_by_tolerance(x_clear, all_x_explored, tolerances)
        )
        self.assertTrue(is_duplicate_by_tolerance(x_dup, all_x_explored, tolerances))

    def test_get_len(self):
        self.assertEqual(get_len([1, 2, 3]), 3)
        self.assertEqual(get_len(4), 1)
        self.assertEqual(get_len("abc"), 1)
