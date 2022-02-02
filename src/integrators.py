#!/usr/bin/env python3

"""n-body-sim: src/integrators
Implementations of different integration methods.
"""
import numpy as np

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
def pefrl(masses, pos, vel, dt, acc):
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


@solver
def rk8(masses, pos, vel, dt, acc):
    """
    Solves N body simulation using an eighth-order Runge-Kutta method based on
    the Butcher tableau provided by: Dormand, Prince. "A reconsideration of
    some embedded Runge-Kutta formulae". 1986.

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

    # fmt: off
    a = np.array(
        (
            (                      0,      0,        0,                          0,                       0,                         0,                         0,                         0,                         0,                       0,                      0, 0, 0),
            (                 1 / 18,      0,        0,                          0,                       0,                         0,                         0,                         0,                         0,                       0,                      0, 0, 0),
            (                 1 / 48, 1 / 16,        0,                          0,                       0,                         0,                         0,                         0,                         0,                       0,                      0, 0, 0),
            (                 1 / 32,      0,   3 / 32,                          0,                       0,                         0,                         0,                         0,                         0,                       0,                      0, 0, 0),
            (                 5 / 16,      0, -75 / 64,                    75 / 64,                       0,                         0,                         0,                         0,                         0,                       0,                      0, 0, 0),
            (                 3 / 80,      0,        0,                     3 / 16,                  3 / 20,                         0,                         0,                         0,                         0,                       0,                      0, 0, 0),
            (   29443841 / 614563906,      0,        0,       77736538 / 692538347,  -28693883 / 1125000000,     23124283 / 1800000000,                         0,                         0,                         0,                       0,                      0, 0, 0),
            (   16016141 / 946692911,      0,        0,       61564180 / 158732637,    22789713 / 633445777,    545815736 / 2771057229,   -180193667 / 1043307555,                         0,                         0,                       0,                      0, 0, 0),
            (   39632708 / 573591083,      0,        0,     -433636366 / 683701615, -421739975 / 2616292301,     100302831 / 723423059,     790204164 / 839813087,    800635310 / 3783071287,                         0,                       0,                      0, 0, 0),
            ( 246121993 / 1340847787,      0,        0, -37695042795 / 15268766246, -309121744 / 1061227803,     -12992083 / 490766935,   6005943493 / 2108947869,    393006217 / 1396673457,    123872331 / 1001029789,                       0,                      0, 0, 0),
            (-1028468189 / 846180014,      0,        0,     8478235783 / 508512852, 1311729495 / 1432422823, -10304129995 / 1701304382, -48777925059 / 3047939560,  15336726248 / 1032824649, -45442868181 / 3398467696,  3065993473 / 597172653,                      0, 0, 0),
            (  185892177 / 718116043,      0,        0,    -3185094517 / 667107341, -477755414 / 1098053517,    -703635378 / 230739211,   5731566787 / 1027545527,    5232866602 / 850066563,   -4093664535 / 808688257, 3962137247 / 1805957418,   65686358 / 487910083, 0, 0),
            (  403863854 / 491063109,      0,        0,    -5068492393 / 434740067,  -411421997 / 543043805,     652783627 / 914296604,   11173962825 / 925320556, -13158990841 / 6184727034,   3936647629 / 1978049680,  -160528059 / 685178525, 248638103 / 1413531060, 0, 0),
        )
    )
    b = np.array((14005451 / 335480064, 0, 0, 0, 0, -59238493 / 1068277825, 181606767 / 758867731, 561292985 / 797845732, -1041891430 / 1371343529, 760417239 / 1151165299, 118820643 / 751138087, -528747749 / 2220607170, 1 / 4))
    # fmt: on

    k = np.zeros((len(masses), 2, len(b)))
    j = np.zeros((len(masses), 2, len(b)))

    for time_idx in range(pos.shape[2] - 1):

        for sub_step in range(len(b)):
            k[:, :, sub_step] = vel[:, :, time_idx] + dt * np.sum(a[sub_step, :] * j, axis=2)
            j[:, :, sub_step] = acc(
                masses, pos[:, :, time_idx] + dt * np.sum(a[sub_step, :] * k, axis=2)
            )

        pos[:, :, time_idx + 1] = pos[:, :, time_idx] + dt * np.sum(b * k, axis=2)
        vel[:, :, time_idx + 1] = vel[:, :, time_idx] + dt * np.sum(b * j, axis=2)

    return pos, vel
