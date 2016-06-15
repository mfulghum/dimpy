"""
Dimensional analysis functions
Copyright 2016, Matthew Fulghum [mfulghum]

15 June 2016 - Initial commit.
"""

import numpy as np

def get_dimensional_set_shape(variables):
    dimensions = np.vstack([variable.dimensions for variable in variables])

    num_variables = dimensions.shape[0]
    num_dimensions = np.sum(np.sum(dimensions != 0, axis=0) != 0)
    num_nonzero = np.sum(dimensions != 0)
    num_groups = num_variables - num_dimensions + int(num_nonzero > 0)

    return int(num_variables), int(num_dimensions), int(num_groups)