#!/usr/bin/env python3

"""n-body-sim: src/integrators
Implementation of a solver that uses the forward Euler method.
"""
import numpy as np

from utils import solver


__author__ = "jeypiti"
__copyright__ = "Copyright 2021, jeypiti"
__credits__ = ["jeypiti"]
__license__ = "MIT"


@solver
def forward_euler(masses, pos, vel, dt):
    """
    Solves N body simulation using the forward Euler method.

    :param masses: List of N masses.
    :param pos: Nearly empty list for time evolution of x & y coordinates for N bodies.
                Only initial conditions should be specified.
    :param vel: Nearly empty list for time evolution of x & y velocities for N bodies.
                Only initial conditions should be specified.
    :param dt: Time step for the simulation.
    :return: Time evolution of x & y coordinates for N bodies.
    """

    for time_idx in range(pos.shape[2] - 1):
        pos[:, :, time_idx + 1] = pos[:, :, time_idx] + dt * vel[:, :, time_idx]
        vel[:, :, time_idx + 1] = vel[:, :, time_idx] + dt * acc(masses, pos[:, :, time_idx])

    return pos, vel


def acc(masses, current_pos):
    """
    Calculate the acc for each body in both
    x & y direction based on the gravitational force.

    :param masses: List of N masses.
    :param current_pos: List of current positions of N bodies.
    :return: Tuple containing a list of N forces in the x direction and a list
             of N forces in the y direction.
    """

    result = np.zeros((len(masses), 2))

    for m1 in range(len(masses)):
        for m2 in range(len(masses)):
            if m1 != m2:
                dist = current_pos[m1, :] - current_pos[m2, :]
                result[m1, :] -= masses[m2] * dist * dist.dot(dist) ** -1.5

    return result
