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
