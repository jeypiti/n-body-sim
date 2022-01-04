#!/usr/bin/env python3

"""n-body-sim: src/simulation
Main file of the simulation. Sets up integration method, initial conditions,
and time steps. Integrates N body problem and displays animation.
"""
from time import perf_counter

import numpy as np

import direct_sum
import barnes_hut
import integrators
from utils import animate


__author__ = "jeypiti"
__copyright__ = "Copyright 2021, jeypiti"
__credits__ = ["jeypiti"]
__license__ = "MIT"


if __name__ == '__main__':
    time_steps = 1000
    t_end = 10
    t, dt = np.linspace(0, t_end, num=time_steps, retstep=True)

    masses = np.array([0.1, 1, 0.1], dtype=float)

    # set initial conditions for position and velocity
    pos = np.array([[-5, -1], [0, 0], [1, -5]])
    vel = np.array([[1, 0], [0, 0], [0, 1]])

    start = perf_counter()
    pos, vel = integrators.perfl(
        masses,
        pos,
        vel,
        time_steps,
        dt,
        direct_sum.acceleration_vec,
    )
    end = perf_counter()
    print(f"Computation for {len(masses)} bodies and {time_steps} time steps took {end - start:.3f} s")

    animate(masses, pos, vel, t)
