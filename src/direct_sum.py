#!/usr/bin/env python3

"""n-body-sim: src/direct_sum
Implementation of the direct sum method of calculating
the acceleration caused by the gravitational force.
"""
import numpy as np


__author__ = "jeypiti"
__copyright__ = "Copyright 2021, jeypiti"
__credits__ = ["jeypiti"]
__license__ = "MIT"


def acceleration(masses, current_pos):
    """
    Calculate the acceleration for each body in both x & y direction
    based on the gravitational force using the direct sum approach.

    :param masses: List of N masses.
    :param current_pos: List of current positions of N bodies.
    :return: An (N, 2) array of accelerations.
    """

    result = np.zeros((len(masses), 2))

    for m1 in range(len(masses)):
        for m2 in range(len(masses)):
            if m1 != m2:
                dist = current_pos[m1, :] - current_pos[m2, :]
                result[m1, :] -= masses[m2] * dist * dist.dot(dist) ** -1.5

    return result
