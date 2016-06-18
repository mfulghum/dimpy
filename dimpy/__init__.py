"""
Dimensioned array extension to numpy
Copyright 2016, Matthew Fulghum [mfulghum]

5 June 2016 - Initial commit.
"""

from array import data, unit, DimensionError
from units import check_dimensionality
from analysis import analyze