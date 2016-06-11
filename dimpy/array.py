"""
Dimensioned array extension to numpy
Copyright 2016, Matthew Fulghum [mfulghum]

5 June 2016 - Initial commit.
9 June 2016 - Added checks for dimensionality.
"""

import numpy as np
import units as unit_check

class data(np.ndarray):
    def __new__(cls, input_array, units='', dimensions=None):
        power, dimensions = unit_check.check_dimensionality(units) if dimensions is None else (0, np.array(dimensions, dtype=np.int8))        
        obj = np.asarray(input_array).view(cls) * 10.0**power
        obj.dimensions = dimensions
        return obj

    def __array_finalize__(self, obj):
        if obj is None: 
            return
        self.dimensions = getattr(obj, 'dimensions', np.zeros(7, dtype=np.int8))

    def __getitem__(self, item):
        return self.__class__(super(data, self).__getitem__(item), dimensions=self.dimensions)

    def __repr__(self):
        return '%s - dimensions: [mass: %d, length: %d, time: %d, current: %d, temperature: %d, amount of substance: %d, luminous intensity: %d]' % \
            (str(np.asfarray(self)), self.dimensions[0], self.dimensions[1], self.dimensions[2], self.dimensions[3], self.dimensions[4], self.dimensions[5], self.dimensions[6])

    def __array_prepare__(self, out_arr, context=None):
        ufunc_type = str(context[0]).split("'")[1]
            
        if ufunc_type in ['add', 'subtract', 'mod', 'fmod', 'remainder',
                          'greater', 'greater_equal', 'equal', 'less_equal', 'less', 'not_equal']:
            # These operations require dimensions to be consistent between operands
            left_operand, right_operand = context[1][:2]
            assert isinstance(left_operand, type(self)) and isinstance(right_operand, type(self))
            assert np.all(left_operand.dimensions == right_operand.dimensions)
        elif ufunc_type in ['sin', 'cos', 'tan',
                            'arcsin', 'arccos', 'arctan', 'arctan2',
                            'sinh', 'cosh', 'tanh',
                            'arcsinh', 'arccosh', 'arctanh',
                            'deg2rad', 'rad2deg',
                            'exp', 'exp2', 'log', 'log2', 'log10',
                            'expm1', 'log1',
                            'logaddexp', 'logaddexp2']:
            # These operations require that the inputs be dimensionless
            operand = context[1][0]
            assert np.all(operand.dimensions == 0)
        elif ufunc_type in ['sqrt']:
            # The result of a sqrt operation must have even dimension powers
            operand = context[1][0]
            assert np.all((operand.dimensions % 2) == 0)
        elif ufunc_type in ['power']:
            # The result of a power operation must result in integer dimension powers
            left_operand, right_operand = context[1][:2]
            assert np.all(not isinstance(right_operand, type(self)) or right_operand.dimensions == 0)
            assert np.all(right_operand == np.round(np.asarray(right_operand)))
            
        return np.ndarray.__array_prepare__(self, out_arr, context)

    def __array_wrap__(self, out_arr, context=None):
        ufunc_type = str(context[0]).split("'")[1]
        
        if ufunc_type in ['multiply', 'divide', 'true_divide', 'floor_divide']:
            # The result should add the dimensions of the two operands together
            left_operand, right_operand = context[1][:2]
            out_arr.dimensions = (left_operand.dimensions if isinstance(left_operand, type(self)) else 0) + \
                                 (right_operand.dimensions if isinstance(right_operand, type(self)) else 0)
        elif ufunc_type in ['power']:
            # The result should have its dimensions multiplied by the exponent
            left_operand, right_operand = context[1][:2]
            out_arr.dimensions *= np.asarray(right_operand)
        elif ufunc_type in ['sqrt']:
            # The result should have halved dimension powers
            out_arr.dimensions /= 2
        elif ufunc_type in ['square']:
            # The result should have doubled dimension powers
            out_arr.dimensions *= 2
                              
        return np.ndarray.__array_wrap__(self, out_arr, context)
