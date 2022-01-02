#!/usr/bin/env python3

"""n-body-sim: src/integrators
Implementations of different integration methods.
"""
from utils import solver


__author__ = "jeypiti, Lvisss"
__copyright__ = "Copyright 2021, jeypiti, Lvisss"
__credits__ = ["jeypiti", "Lvisss"]
__license__ = "MIT"


@solver
def forward_euler(masses, pos, vel, dt, acc_func):
    """
    Solves N body simulation using the forward Euler method.

    :param masses: List of N masses.
    :param pos: Nearly empty list for time evolution of x & y coordinates for N bodies.
                Only initial conditions should be specified.
    :param vel: Nearly empty list for time evolution of x & y velocities for N bodies.
                Only initial conditions should be specified.
    :param dt: Time step for the simulation.
    :param acc_func: Callable that calculates the acceleration acting on the bodies.
                     Takes an (N,) array of masses and an (N, 2) array of current
                     positions. Returns an (N, 2) array of accelerations.
    :return: Time evolution of x & y coordinates for N bodies.
    """

    for time_idx in range(pos.shape[2] - 1):
        pos[:, :, time_idx + 1] = pos[:, :, time_idx] + dt * vel[:, :, time_idx]
        vel[:, :, time_idx + 1] = vel[:, :, time_idx] + dt * acc_func(masses, pos[:, :, time_idx])

    return pos, vel


@solver
def leapfrog(masses, pos, vel, dt, acc_func):
    """
    Solves N body simulation using the leapfrog method.

    :param masses: List of N masses.
    :param pos: Nearly empty list for time evolution of x & y coordinates for N bodies.
                Only initial conditions should be specified.
    :param vel: Nearly empty list for time evolution of x & y velocities for N bodies.
                Only initial conditions should be specified.
    :param dt: Time step for the simulation.
    :param acc_func: Callable that calculates the acceleration acting on the bodies.
                     Takes an (N,) array of masses and an (N, 2) array of current
                     positions. Returns an (N, 2) array of accelerations.
    :return: Time evolution of x & y coordinates for N bodies.
    """

    for time_idx in range(pos.shape[2] - 1):
        # update next time step with 1/2 drift
        pos[:, :, time_idx + 1] = pos[:, :, time_idx] + 0.5 * dt * vel[:, :, time_idx]

        # use 1/2 drift step for full kick step
        vel[:, :, time_idx + 1] = vel[:, :, time_idx] + dt * acc_func(masses, pos[:, :, time_idx + 1])

        # another 1/2 drift
        pos[:, :, time_idx + 1] = pos[:, :, time_idx + 1] + 0.5 * dt * vel[:, :, time_idx + 1]

    return pos, vel
