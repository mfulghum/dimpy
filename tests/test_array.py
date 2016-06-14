import unittest
import numpy as np

import dimpy
import dimpy.units

class array_tests(unittest.TestCase):
    def test_repr(self):
        # Check that the array repr matches intended output
        self.assertEquals(repr(dimpy.data([1.0, 2.0, 3.0], units='N')), '[ 1.  2.  3.] - dimensions: [mass: 1, length: 1, time: -2, current: 0, temperature: 0, amount of substance: 0, luminous intensity: 0]')

    def test_powers(self):
        # Check that arrays scale properly to the native internal representation
        self.assertTrue(np.all(dimpy.data([1.0, 2.0, 3.0], units='mm') == dimpy.data([1e-3, 2e-3, 3e-3], units='m')))

    def test_dimensional_homogeneity(self):
        # Check that you cannot add dissimilar dimensions
        with self.assertRaises(dimpy.DimensionError):
            dimpy.data([1.0], units='m') + dimpy.data([1.0], units='s')

        # Check that you can't take the sine of a dimensioned value
        with self.assertRaises(dimpy.DimensionError):
            np.sin(dimpy.data([1.0], units='m'))

        # Check that you can't take the square root of an odd dimension
        with self.assertRaises(dimpy.DimensionError):
            np.sqrt(dimpy.data([1.0], units='L'))

        # Check that you can't raise a value to a non-integer value
        with self.assertRaises(dimpy.DimensionError):
            dimpy.data([1.0], units='m') ** 1.1

    def test_dimension_arithmetic(self):
        # Check that multiplication results in summation of dimensions
        mass = dimpy.data([1.0], units='kg')
        acceleration = dimpy.data([1.0], units='m/s^2')
        force = dimpy.data([1.0], units='N')

        self.assertTrue(mass * acceleration == force)

        # Check that taking the square root halves the dimensions
        area = dimpy.data([100.0], units='m^2')
        side = dimpy.data([10.0], units='m')

        self.assertTrue(np.sqrt(area) == side)
