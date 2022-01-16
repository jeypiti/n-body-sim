#!/usr/bin/env python3

"""n-body-sim: src/init_cond
Set of initial conditions that can be used for the simulation.
"""
import numpy as np


__author__ = "jeypiti"
__copyright__ = "Copyright 2022, jeypiti"
__credits__ = ["jeypiti"]
__license__ = "MIT"

solar = 1.9885e30
mercury = 3.3011e23
venus = 4.8675e24
earth = 5.97237e24
mars = 6.4171e23
jupiter = 1.8982e27
saturn = 5.6834e26
neptune = 1.02413e26
uranus = 8.6810e25
pluto = 1.303e22


def generate_planetary_system(bodies, max_mass=1e-3, seed=None):
    """
    Randomly generates initial conditions for a planetary system. Depending on
    the exact starting positions, numerical errors in the simulation may lead
    to problems, e.g. bodies being ejected from the system. The chance of this
    happening increases with the number of bodies in the system.
    The system will be spinning anticlockwise.

    :param bodies: Number of bodies in the system.
    :param max_mass: Maximum mass of the outer bodies. The central body has mass 1.
    :param seed: Seed for the pseudorandom number generator.
    :return: Vales for (n,) masses, (n, 2) positions, and (n, 2) velocities.
    """

    rng = np.random.default_rng(seed)

    masses = rng.uniform(0, max_mass, (bodies,))

    # generate polar coordinates & convert to cartesian
    theta = rng.uniform(0, 2 * np.pi, (bodies,))
    r = rng.gamma(7.5, 1, (bodies,)) * bodies ** 0.8
    pos = np.vstack((r * np.cos(theta), r * np.sin(theta))).T

    # calculate magnitude of velocity under the approximation of a two-body system
    v = np.sqrt(1 / r)

    vel = np.vstack((-v * np.sin(theta), v * np.cos(theta))).T

    # overwrite first body with heavy central mass
    masses[0] = 1
    pos[0, :] = vel[0, :] = 0

    return masses, pos, vel


# modeled after our solar system
solar_system = (
    np.array((solar, mercury, venus, earth, mars, jupiter, saturn, uranus, pluto)) / solar,
    np.array(
        (
            (0, 0),
            (0, 1.29),
            (0, -2.4),
            (-2.35, 2.35),
            (3.58, -3.58),
            (12.21, 12.21),
            (-22.44, -22.44),
            (-45.07, 45.07),
            (99.03, 13.92),
        )
    ),
    np.array(
        (
            (0, 0),
            (0.88, 0),
            (-0.645, 0),
            (0.388, 0.388),
            (-0.314, -0.314),
            (0.17, -0.17),
            (-0.125, 0.125),
            (0.0886, 0.0886),
            (0.014, -0.099),
        )
    ),
)

# solar system reduced to four bodies
small_solar_system = (
    np.array((solar, mercury, venus, uranus)) / solar,
    np.array(((0, 0), (0, 1.29), (0, -2.4), (-45.07, 45.07))),
    np.array(((0, 0), (0.88, 0), (-0.645, 0), (0.0886, 0.0886))),
)

# periodic figure eight structure by:
# Chenciner, Montgomery. "A remarkable periodic solution of the
# three-body problem in the case of equal masses". 2000.
figure_eight = (
    np.array((1.0, 1.0, 1.0)),
    np.array(
        (
            (-0.97000436, 0.24308753),
            (0.0, 0.0),
            (0.97000436, -0.24308753),
        )
    ),
    np.array(
        (
            (0.93240737 / 2, 0.86473146 / 2),
            (-0.93240737, -0.86473146),
            (0.93240737 / 2, 0.86473146 / 2),
        )
    ),
)

random_planetary_system = generate_planetary_system(20, seed=31415)

# periodic initial conditions by:
# Xiaoming Li, Yipeng Jing, Shijun Liao. "The 1223 new periodic orbits of
# planar three-body problem with unequal mass and zero angular momentum". 2017.
three_body_periodic1 = (
    np.array((1.0, 1.0, 0.5)),
    np.array(
        (
            (-1.0, 0.0),
            (1.0, 0.0),
            (0.0, 0.0),
        )
    ),
    np.array(
        (
            (0.2009656237, 0.2431076328),
            (0.2009656237, 0.2431076328),
            (-4 * 0.2009656237, -4 * 0.2431076328),
        )
    ),
)

three_body_periodic2 = (
    np.array((1.0, 1.0, 2.0)),
    np.array(
        (
            (-1.0, 0.0),
            (1.0, 0.0),
            (0.0, 0.0),
        )
    ),
    np.array(
        (
            (0.6649107583, 0.8324167864),
            (0.6649107583, 0.8324167864),
            (-0.6649107583, -0.8324167864),
        )
    ),
)

# periodic free-fall initial conditions by:
# Xiaoming Li, Shijun Liao. "Collisionless periodic
# orbits in the free-fall three body problem". 2018.
free_fall_periodic1 = (
    np.array((1.0, 0.8, 0.8)),
    np.array(
        (
            (-0.5, 0.0),
            (0.5, 0.0),
            (0.0009114239, 0.3019805958),
        )
    ),
    np.array(
        (
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        )
    ),
)

free_fall_periodic2 = (
    np.array((1.0, 0.8, 0.4)),
    np.array(
        (
            (-0.5, 0.0),
            (0.5, 0.0),
            (0.1204686367, 0.3718569619),
        )
    ),
    np.array(
        (
            (0.0, 0.0),
            (0.0, 0.0),
            (0.0, 0.0),
        )
    ),
)
