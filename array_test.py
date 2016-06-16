# -*- coding: utf-8 -*-
"""
Created on Sun Jun 05 17:56:27 2016

@author: Matthew Fulghum
"""

import numpy as np
import dimpy

vel = dimpy.data([1.0, 2.0, 3.0], units='mm/s')
area = dimpy.data([20.0], units='m^2')

mass = dimpy.data([1.0], units='kg')
acceleration = dimpy.data([1.0], units='m/s^2')
force = dimpy.data([1.0], units='N')

length = dimpy.data([1.0], units='cm')
volume = dimpy.data([1.0], units='mL')

# Pendulum model
T = dimpy.unit('s')
M = dimpy.unit('kg')
L = dimpy.unit('m')
g = dimpy.unit('m/s^2')

variables = [T, M, L, g]
dimensions = np.vstack([variable.dimensions for variable in variables])
print(dimpy.analysis.get_dimensional_set_shape(variables))

# Black body model
u = dimpy.unit('kg/(m^2-s^2)')
wavelength = dimpy.unit('m')
h = dimpy.unit('kg-m^2/s')
c = dimpy.unit('m/s')
kB = dimpy.unit('kg-m^2/(s^2-K)')
T = dimpy.unit('K')


variables = [u, wavelength, h, c, kB, T]
dimensions = np.vstack([variable.dimensions for variable in variables])
print(dimensions)
print(dimpy.analysis.get_dimensional_set_shape(variables))

num_variables, num_dimensions, num_groups = dimpy.analysis.get_dimensional_set_shape(variables)
#permutations = [[(x+y) % num_variables for x in range(num_variables)][num_groups:] for y in range(num_variables)]

permutations = list({tuple(set(value[num_groups:])) for value in itertools.permutations(range(num_variables))})
dimensions = np.vstack([variable.dimensions for variable in variables])
active_dimensions = np.where(np.sum(dimensions != 0, axis=0) != 0)[0]

for permutation in permutations:
    dependent_matrix = dimensions[np.ix_(permutation, active_dimensions)]
    try:
        assert dependent_matrix.shape[0] == dependent_matrix.shape[1]
        assert np.linalg.det(dependent_matrix) != 0
        assert np.linalg.matrix_rank(dependent_matrix) == dependent_matrix.shape[0]
    except:
        continue
    
    print('permutation: %s' % repr(permutation))
    print(dependent_matrix)
    print('rank: %d' % np.linalg.matrix_rank(dependent_matrix.T))
    print('determinant: %d' % np.linalg.det(dependent_matrix.T))
    print('')


variables = [u, wavelength, h, c]
dimensions = np.vstack([variable.dimensions for variable in variables])
dimensional_set = np.where(np.sum(dimensions != 0, axis=0) != 0)
A = dimensions[:,dimensional_set[0]]
#print(np.linalg.inv(A))
#print(np.linalg.det(A))
print(np.linalg.matrix_rank(A))

variables = [wavelength, h, c, kB]
dimensions = np.vstack([variable.dimensions for variable in variables])
dimensional_set = np.where(np.sum(dimensions != 0, axis=0) != 0)
A = dimensions[:,dimensional_set[0]]
print(np.linalg.inv(A))
#print(np.linalg.det(A))
print(np.linalg.matrix_rank(A))

variables = [h, c, kB, T]
dimensions = np.vstack([variable.dimensions for variable in variables])
dimensional_set = np.where(np.sum(dimensions != 0, axis=0) != 0)
A = dimensions[:,dimensional_set[0]]
print(np.linalg.inv(A))
#print(np.linalg.det(A))
#print(np.linalg.matrix_rank(A))

variables = [c, kB, T, u]
dimensions = np.vstack([variable.dimensions for variable in variables])
dimensional_set = np.where(np.sum(dimensions != 0, axis=0) != 0)
A = dimensions[:,dimensional_set[0]]
print(np.linalg.inv(A))
#print(np.linalg.det(A))
#print(np.linalg.matrix_rank(A))

variables = [kB, T, u, wavelength]
dimensions = np.vstack([variable.dimensions for variable in variables])
dimensional_set = np.where(np.sum(dimensions != 0, axis=0) != 0)
A = dimensions[:,dimensional_set[0]]
print(np.linalg.inv(A))
#print(np.linalg.det(A))
#print(np.linalg.matrix_rank(A))

variables = [T, u, wavelength, h]
dimensions = np.vstack([variable.dimensions for variable in variables])
dimensional_set = np.where(np.sum(dimensions != 0, axis=0) != 0)
A = dimensions[:,dimensional_set[0]]
print(np.linalg.inv(A))
#print(np.linalg.det(A))
#print(np.linalg.matrix_rank(A))