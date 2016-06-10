"""
Dimensioned array extension to numpy
Copyright 2016, Matthew Fulghum [mfulghum]

5 June 2016 - Initial commit.
"""

import re

import numpy as np
import units as unit_check

class data(np.ndarray):
    def __new__(cls, input_array, units='1'):
        obj = np.asarray(input_array).view(cls)
        obj.units = units
        obj.power, obj.dimensions = unit_check.check_dimensionality(units)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.units = getattr(obj, 'units', '1')
        self.power = getattr(obj, 'power', 0)
        self.dimensions = getattr(obj, 'dimensions', np.zeros(7, dtype=np.int8))

    def __repr__(self):
        return '%s - units: %s, dimensions: %s' % (self.__str__(), self.units, self.dimensions)

    def __array_prepare__(self, out_arr, context=None):
        ufunc_type = str(context[0]).split("'")[1]
        num_inputs = context[0].nin

        if ufunc_type in ['add', 'subtract', 'logaddexp', 'logaddexp2',
                          'greater', 'greater_equal', 'equal', 'less_equal', 'less', 'not_equal']:
            assert np.all(context[1][0].dimensions == context[1][1].dimensions)
        elif ufunc_type in ['sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan']:
            assert np.all(context[1][0].dimensions == 0)

        print('In __array_prepare__:')
        print('   ufunc: %s' % ufunc_type)
        print('   data: %s' % str(context[1]))
        # then just call the parent
        return np.ndarray.__array_prepare__(self, out_arr, context)

    def __array_wrap__(self, out_arr, context=None):
        print('In __array_wrap__:')
        print('   self is %s' % repr(self))
        print('   arr is %s' % repr(out_arr))
        print('   context is %s' % str(context))
        # then just call the parent
        return np.ndarray.__array_wrap__(self, out_arr, context)
