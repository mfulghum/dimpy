"""
Dimensioned array extension to numpy
Copyright 2016, Matthew Fulghum [mfulghum]

5 June 2016 - Initial commit.
9 June 2016 - Added checks for dimensionality.
"""

import re

import numpy as np
import units as unit_check

class data(np.ndarray):
    def __new__(cls, input_array, units=''):
        obj = np.asarray(input_array).view(cls)
        obj.units = units
        obj.power, obj.dimensions = unit_check.check_dimensionality(units)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.units = getattr(obj, 'units', '')
        self.power = getattr(obj, 'power', 0)
        self.dimensions = getattr(obj, 'dimensions', np.zeros(7, dtype=np.int8))

    def __getitem__(self, item):
        return self.__class__(super(data, self).__getitem__(item), units=self.units)

    def __repr__(self):
        return '%s - units: %s, dimensions: %s' % (self.__str__(), self.units, self.dimensions)

    def __array_prepare__(self, out_arr, context=None):
        ufunc_type = str(context[0]).split("'")[1]
        num_inputs = context[0].nin

        if ufunc_type in ['add', 'subtract', 'logaddexp', 'logaddexp2', 'mod', 'fmod', 'remainder',
                          'greater', 'greater_equal', 'equal', 'less_equal', 'less', 'not_equal']:
            # These operations require dimensions to be consistent between operands
            assert np.all(context[1][0].dimensions == context[1][1].dimensions)
        elif ufunc_type in ['sin', 'cos', 'tan',
                            'arcsin', 'arccos', 'arctan', 'arctan2',
                            'sinh', 'cosh', 'tanh',
                            'arcsinh', 'arccosh', 'arctanh',
                            'deg2rad', 'rad2deg',
                            'exp', 'exp2', 'log', 'log2', 'log10',
                            'expm1', 'log1']:
            # These operations require that the inputs be dimensionless
            assert np.all(context[1][0].dimensions == 0)
        elif ufunc_type in ['sqrt']:
            # The result of a sqrt operation must have integer dimension powers
            assert np.all((context[1][0].dimensions % 2) == 0)
        elif ufunc_type in ['power']:
            # The result of a power operation must result in integer dimension powers
            assert np.all(not isinstance(context[1][1], type(self)) or context[1][1].dimensions == 0)
            assert np.all(context[1][1] == np.round(np.asarray(context[1][1])))
            
        return np.ndarray.__array_prepare__(self, out_arr, context)

    def __array_wrap__(self, out_arr, context=None):
        print('In __array_wrap__:')
        print('   self is %s' % repr(self))
        print('   arr is %s' % repr(out_arr))
        print('   context is %s' % str(context))
        # then just call the parent
        return np.ndarray.__array_wrap__(self, out_arr, context)
