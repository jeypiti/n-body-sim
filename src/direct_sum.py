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
    Calculates the acceleration for each body in both x & y direction
    based on the gravitational force using the direct sum approach.

    :param masses: A (N,) array of masses.
    :param current_pos: A (N, 2) array of current positions of all bodies.
    :return: A (N, 2) array of accelerations.
    """

    result = np.zeros((len(masses), 2))

    for m1 in range(len(masses)):
        for m2 in range(len(masses)):
            if m1 != m2:
                dist = current_pos[m1, :] - current_pos[m2, :]
                result[m1, :] -= masses[m2] * dist * dist.dot(dist) ** -1.5

    return result


def acceleration_vec(masses, current_pos):
    """
    Calculates the acceleration for each body in both x & y direction based
    on the gravitational force using a vectorized direct sum approach.

    Implementation inspired by Philip Mocz:
    https://github.com/pmocz/nbody-python

    :param masses: A (N,) array of masses.
    :param current_pos: A (N, 2) array of current positions of all bodies.
    :return: A (N, 2) array of accelerations.
    """

    # extract x & y coordinates to a (N, 1) array
    x = current_pos[:, 0:1]
    y = current_pos[:, 1:2]

    # matrices that store pairwise body distances
    dx = x.T - x
    dy = y.T - y

    # calculate r^-3 factor
    # mask exponentiation to avoid division by zero
    factor = dx ** 2 + dy ** 2
    np.float_power(factor, -1.5, where=factor != 0, out=factor)

    # calculate acceleration in x & y direction
    a_x = (dx * factor) @ masses
    a_y = (dy * factor) @ masses

    # pack a_x and a_y into desired data structure
    return np.vstack((a_x, a_y)).T
