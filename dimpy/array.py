"""
Dimensioned array extension to numpy
Copyright 2016, Matthew Fulghum [mfulghum]

5 June 2016 - Initial commit.
"""

import numpy as np
import units as unit_check

class array(np.ndarray):
    def __new__(cls, input_array, units=None):
        obj = np.asarray(input_array).view(cls)
        obj.units = units
        obj.power, obj.dimensions = unit_check.check_dimensionality(units)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.units = getattr(obj, 'units', '1')
        self.power = getattr(obj, 'power', 0)
        self.dimensions = getattr(obj, 'dimensions', np.zeros(7, dtype=np.int8))