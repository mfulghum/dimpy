"""
Dimensional analysis functions
Copyright 2016, Matthew Fulghum [mfulghum]

15 June 2016 - Initial commit.
17 June 2016 - Add a function to find all valid dimensional sets
"""

import itertools
import numpy as np

def get_dimensional_set_shape(variables):
    dimensions = np.vstack([variable.dimensions for variable in variables])

    num_variables = dimensions.shape[0]
    num_dimensions = np.sum(np.sum(dimensions != 0, axis=0) != 0)
    num_groups = num_variables - np.linalg.matrix_rank(dimensions.T)

    return int(num_variables), int(num_dimensions), int(num_groups)

def analyze(variables):
    num_variables, num_dimensions, num_groups = get_dimensional_set_shape(variables)
    permutations = list({tuple(set(value[num_groups:])) for value in itertools.permutations(range(num_variables))})
    dimensions = np.vstack([variable.dimensions for variable in variables])
    active_dimensions = np.where(np.sum(dimensions != 0, axis=0) != 0)[0]

    valid_sets = []

    for permutation in permutations:
        dependent_variables = tuple(set(range(num_variables)) - set(permutation))
        independent_variables = tuple(permutation)

        dependent_matrix = dimensions[np.ix_(dependent_variables, active_dimensions)]
        independent_matrix = dimensions[np.ix_(independent_variables, active_dimensions)]

        try:
            assert independent_matrix.shape[0] == independent_matrix.shape[1]
            assert np.linalg.det(independent_matrix) != 0
            assert np.linalg.matrix_rank(independent_matrix) == independent_matrix.shape[0]
        except:
            continue

        pi_matrix = np.vstack(
            [np.eye(num_groups), -np.linalg.inv(np.matrix(independent_matrix.T)) * np.matrix(dependent_matrix.T)])

        variable_order = dependent_variables + independent_variables
        idx = tuple(
            [value[1] for value in sorted(zip(variable_order, range(len(variable_order))), key=lambda value: value[0])])
        pi_matrix = pi_matrix[np.ix_(idx, range(num_groups))]

        if not np.all(dimensions.T * pi_matrix == 0):
            continue

        u, s, vh = np.linalg.svd(pi_matrix.T)
        tol = max(1e-13, 1e-12 * s[0])
        rank = (s >= tol).sum()
        nullspace = vh[rank:].conj().T

        in_valid_sets = False
        for (_, valid_rank, valid_nullspace) in valid_sets:
            in_valid_sets |= (rank == valid_rank) and np.allclose(nullspace, valid_nullspace)
        if not in_valid_sets:
            valid_sets.append((pi_matrix, rank, nullspace))
            
    return [np.array(valid_set[0]) for valid_set in valid_sets]
