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
def forward_euler(masses, pos, vel, dt, acc):
    """
    Solves N body simulation using the forward Euler method.

    :param masses: List of N masses.
    :param pos: Nearly empty list for time evolution of x & y coordinates for N bodies.
                Only initial conditions should be specified.
    :param vel: Nearly empty list for time evolution of x & y velocities for N bodies.
                Only initial conditions should be specified.
    :param dt: Time step for the simulation.
    :param acc: Callable that calculates the acceleration acting on the bodies.
                Takes an (N,) array of masses and an (N, 2) array of current
                positions. Returns an (N, 2) array of accelerations.
    :return: Time evolution of x & y coordinates for N bodies.
    """

    for time_idx in range(pos.shape[2] - 1):
        pos[:, :, time_idx + 1] = pos[:, :, time_idx] + dt * vel[:, :, time_idx]
        vel[:, :, time_idx + 1] = vel[:, :, time_idx] + dt * acc(masses, pos[:, :, time_idx])

    return pos, vel


@solver
def leapfrog(masses, pos, vel, dt, acc):
    """
    Solves N body simulation using the leapfrog method.

    :param masses: List of N masses.
    :param pos: Nearly empty list for time evolution of x & y coordinates for N bodies.
                Only initial conditions should be specified.
    :param vel: Nearly empty list for time evolution of x & y velocities for N bodies.
                Only initial conditions should be specified.
    :param dt: Time step for the simulation.
    :param acc: Callable that calculates the acceleration acting on the bodies.
                Takes an (N,) array of masses and an (N, 2) array of current
                positions. Returns an (N, 2) array of accelerations.
    :return: Time evolution of x & y coordinates for N bodies.
    """

    for time_idx in range(pos.shape[2] - 1):
        # update next time step with 1/2 drift
        pos[:, :, time_idx + 1] = pos[:, :, time_idx] + 0.5 * dt * vel[:, :, time_idx]

        # use 1/2 drift step for full kick step
        vel[:, :, time_idx + 1] = vel[:, :, time_idx] + dt * acc(masses, pos[:, :, time_idx + 1])

        # another 1/2 drift
        pos[:, :, time_idx + 1] = pos[:, :, time_idx + 1] + 0.5 * dt * vel[:, :, time_idx + 1]

    return pos, vel


@solver
def perfl(masses, pos, vel, dt, acc):
    """
    Solves N body simulation using the position extended Forest-Ruth-like
    integrator described by:
    Omelyan, Mryglod, Folk. "Optimized Forest-Ruth- and Suzuki-like algorithms
    for integration of motion in many-body systems". 2008.
    URL: https://arxiv.org/pdf/cond-mat/0110585.pdf.

    :param masses: List of N masses.
    :param pos: Nearly empty list for time evolution of x & y coordinates for N bodies.
                Only initial conditions should be specified.
    :param vel: Nearly empty list for time evolution of x & y velocities for N bodies.
                Only initial conditions should be specified.
    :param dt: Time step for the simulation.
    :param acc: Callable that calculates the acceleration acting on the bodies.
                Takes an (N,) array of masses and an (N, 2) array of current
                positions. Returns an (N, 2) array of accelerations.
    :return: Time evolution of x & y coordinates for N bodies.
    """

    xi = 1.786178958448091e-1
    lbd = -2.123418310626054e-1
    chi = -6.626458266981849e-2

    for i in range(pos.shape[2] - 1):
        pos1 = pos[:, :, i] + xi * dt * vel[:, :, i]
        vel1 = vel[:, :, i] + (0.5 - lbd) * dt * acc(masses, pos1)

        pos2 = pos1 + chi * dt * vel1
        vel2 = vel1 + lbd * dt * acc(masses, pos2)

        pos3 = pos2 + (1 - 2 * (chi + xi)) * dt * vel2
        vel3 = vel2 + lbd * dt * acc(masses, pos3)

        pos4 = pos3 + chi * dt * vel3

        vel[:, :, i + 1] = vel3 + (0.5 - lbd) * dt * acc(masses, pos4)
        pos[:, :, i + 1] = pos4 + xi * dt * vel[:, :, i + 1]

    return pos, vel
