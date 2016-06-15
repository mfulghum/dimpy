"""
Unit tests for the dimension checks
Copyright 2016, Matthew Fulghum [mfulghum]

12 June 2016 - Initial commit.
"""

import unittest
import numpy as np

import dimpy.units

class unit_tests(unittest.TestCase):
    def test_dimensionless(self):
        # Check that dimensionless values have a power of 0 and zero dimensions
        power, dimensions = dimpy.units.check_dimensionality('')
        self.assertEqual(power, 0)
        self.assertEqual(np.all(dimensions == 0), True)

    def test_powers(self):
        # Check with a straightforward unit
        self.assertEqual(dimpy.units.check_dimensionality('EPa')[0], 18)
        self.assertEqual(dimpy.units.check_dimensionality('PPa')[0], 15)
        self.assertEqual(dimpy.units.check_dimensionality('TPa')[0], 12)
        self.assertEqual(dimpy.units.check_dimensionality('GPa')[0], 9)
        self.assertEqual(dimpy.units.check_dimensionality('MPa')[0], 6)
        self.assertEqual(dimpy.units.check_dimensionality('kPa')[0], 3)
        self.assertEqual(dimpy.units.check_dimensionality('Pa')[0], 0)
        self.assertEqual(dimpy.units.check_dimensionality('dPa')[0], -1)
        self.assertEqual(dimpy.units.check_dimensionality('cPa')[0], -2)
        self.assertEqual(dimpy.units.check_dimensionality('mPa')[0], -3)
        self.assertEqual(dimpy.units.check_dimensionality('uPa')[0], -6)
        self.assertEqual(dimpy.units.check_dimensionality('nPa')[0], -9)
        self.assertEqual(dimpy.units.check_dimensionality('pPa')[0], -12)
        self.assertEqual(dimpy.units.check_dimensionality('fPa')[0], -15)
        self.assertEqual(dimpy.units.check_dimensionality('aPa')[0], -18)

        # Check with kilograms (power offset by 3)
        self.assertEqual(dimpy.units.check_dimensionality('Eg')[0], 15)
        self.assertEqual(dimpy.units.check_dimensionality('Pg')[0], 12)
        self.assertEqual(dimpy.units.check_dimensionality('Tg')[0], 9)
        self.assertEqual(dimpy.units.check_dimensionality('Gg')[0], 6)
        self.assertEqual(dimpy.units.check_dimensionality('Mg')[0], 3)
        self.assertEqual(dimpy.units.check_dimensionality('kg')[0], 0)
        self.assertEqual(dimpy.units.check_dimensionality('g')[0], -3)
        self.assertEqual(dimpy.units.check_dimensionality('dg')[0], -4)
        self.assertEqual(dimpy.units.check_dimensionality('cg')[0], -5)
        self.assertEqual(dimpy.units.check_dimensionality('mg')[0], -6)
        self.assertEqual(dimpy.units.check_dimensionality('ug')[0], -9)
        self.assertEqual(dimpy.units.check_dimensionality('ng')[0], -12)
        self.assertEqual(dimpy.units.check_dimensionality('pg')[0], -15)
        self.assertEqual(dimpy.units.check_dimensionality('fg')[0], -18)
        self.assertEqual(dimpy.units.check_dimensionality('ag')[0], -21)

    def test_units(self):
        # Check that the dimensions of a newton are [M][L][T]^-2
        self.assertTrue(np.all(dimpy.units.check_dimensionality('kg-m/s^2')[1] == np.array([1, 1, -2, 0, 0, 0, 0])))

                               # Check that units in the divisor have negative powers.
        self.assertTrue(np.all(dimpy.units.check_dimensionality('1/m')[1] == dimpy.units.check_dimensionality('m^-1')[1]))

        # Check that derived units match their base unit representation. In this case, newtons = kg-m/s^2
        self.assertTrue(np.all(dimpy.units.check_dimensionality('kg-m/s^2')[1] == dimpy.units.check_dimensionality('N')[1]))
