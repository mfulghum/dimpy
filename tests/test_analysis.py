"""
Unit tests for dimensional analysis functions
Copyright 2016, Matthew Fulghum [mfulghum]

15 June 2016 - Initial commit.
"""

import unittest
import numpy as np

import dimpy

class analysis_tests(unittest.TestCase):
    def test_set_shape(self):
        # Define some arbitrary data
        current = dimpy.data([1.0, 2.0, 3.0], units='A')
        volume = dimpy.data([1.0], units='mL')
        mass = dimpy.data([1.0], units='kg')
        acceleration = dimpy.data([1.0], units='m/s^2')
        force = dimpy.data([1.0], units='N')

        # Check that the dimensional set shapes are correct
        self.assertEquals(dimpy.analysis.get_dimensional_set_shape([mass, acceleration, force]), (3, 3, 1))
        self.assertEquals(dimpy.analysis.get_dimensional_set_shape([mass, acceleration, force, current]), (4, 4, 1))
        self.assertEquals(dimpy.analysis.get_dimensional_set_shape([mass, acceleration, force, volume]), (4, 3, 1))
        self.assertEquals(dimpy.analysis.get_dimensional_set_shape([force, current]), (2, 4, 0))
        self.assertEquals(dimpy.analysis.get_dimensional_set_shape([force, volume]), (2, 3, 0))