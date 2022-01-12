#!/usr/bin/env python3

"""n-body-sim: src/utils
File holding various utilities.
"""
from functools import partial, wraps
from multiprocessing import Pool
from time import perf_counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation as animation


__author__ = "jeypiti"
__copyright__ = "Copyright 2021, jeypiti"
__credits__ = ["jeypiti"]
__license__ = "MIT"


mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.sans-serif"] = "Latin Modern Roman"


def solver(func):
    """
    Decorator for solver functions. Handles basic errors
    and sets up data structures for the simulation.

    :param func: Solver function.
    :return: Wrapped solver function.
    """

    @wraps(func)
    def wrapper(masses, init_pos, init_vel, time_steps, dt, acc_func):

        if not init_pos.shape[0] == init_vel.shape[0] == len(masses):
            raise ValueError(
                "init_pos and init_vel must specify initial conditions for the same number of"
                f" bodies as provided in masses. You defined masses for {len(masses)} bodies but"
                f" gave initial conditions for {init_pos.shape[0]} and {init_vel.shape[0]} bodies"
                " respectively."
            )

        if not init_pos.shape[1] == init_vel.shape[1] == 2:
            raise ValueError(
                "init_pos and init_vel must specify initial conditions in two dimensions. You gave"
                f" initial conditions for {init_pos.shape[1]} and {init_vel.shape[1]} dimensions"
                " respectively."
            )

        # set up arrays for position and velocity
        # axis 0 describes the masses
        # axis 1 describes the coordinates, i.e. 0 -> x, 1 -> y
        # axis 2 describes the time evolution of the coordinates
        pos = np.zeros((len(masses), 2, time_steps))
        vel = np.zeros((len(masses), 2, time_steps))

        pos[:, :, 0] = init_pos
        vel[:, :, 0] = init_vel

        return func(masses, pos, vel, dt, acc_func)

    return wrapper


def animate(masses, pos, vel, times, duration=3, max_frame_rate=60, save_to_path=None):
    """
    Produces an animation of a given array of positions.
    Optionally saves the animation to a video file if a path is specified.

    :param masses: Array of masses.
    :param pos: Array of positions over time.
    :param vel: Array of velocities over time.
    :param times: Array of time values.
    :param duration: Duration of the animation in seconds.
    :param max_frame_rate: Maximum frame rate at which the animation
                           should be displayed or saved.
    :param save_to_path: If given, saves the animation to this file path.
    """

    start = perf_counter()
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(13, 5), tight_layout=True)
    times = times / times[-1]  # map time to interval [0, 1]

    # prepare a list of artists for animation
    bodies = [ax1.plot([], [], marker="o")[0] for _ in range(len(masses))]
    energy_points = [ax2.plot([], [], marker=".")[0] for _ in range(3)]

    # N body visualization
    pad = 0.03  # padding of the plot as a percentage of the maximum extent of the simulation

    x_min, x_max = np.min(pos[:, 0, :]), np.max(pos[:, 0, :])
    x_range = x_max - x_min
    ax1.set_xlim(x_min - pad * x_range, x_max + pad * x_range)

    y_min, y_max = np.min(pos[:, 1, :]), np.max(pos[:, 1, :])
    y_range = y_max - y_min
    ax1.set_ylim(y_min - pad * y_range, y_max + pad * y_range)

    # ax1.set_ylim(-1, 1)
    # ax1.set_aspect("equal")
    ax1.set_xlabel(r"$x\,/\,\mathrm{a.u.}$")
    ax1.set_ylabel(r"$y\,/\,\mathrm{a.u.}$")

    # energy plot
    kin_energy, pot_energy, tot_energy = calculate_energy(masses, pos, vel)

    plt.gca().set_prop_cycle(None)  # reset color cycle
    ax2.plot(times, kin_energy, label=r"$E_\mathrm{kin}$")
    ax2.plot(times, pot_energy, label=r"$E_\mathrm{pot}$")
    ax2.plot(times, tot_energy, label=r"$E_\mathrm{tot}$")

    ax2.set_xlim(0, 1)
    ax2.set_xlabel(r"$t\,/\,\mathrm{a.u.}$")
    ax2.set_ylabel(r"$E\,/\,\mathrm{a.u.}$")
    ax2.legend()

    # generate animation
    frame_count = int(min(pos.shape[2], duration * max_frame_rate))
    frame_step = pos.shape[2] / frame_count
    frame_rate = frame_count / duration
    frame_time = max(1.0, 1000 / frame_rate)

    print(
        f"Rendering {frame_count * frame_time / 1000:.1f}s animation @ {frame_rate:.1f} FPS"
        f" ({frame_time:.1f} ms per frame)\nShowing {frame_count} frames from a total of"
        f" {pos.shape[2]:,} time steps (one frame every {frame_step:,.1f} steps)"
    )

    def get_frame(frame_idx):
        time_idx = int(frame_idx * frame_step)

        # update bodies
        for mass_idx, body in enumerate(bodies):
            body.set_data(pos[mass_idx, :, time_idx])

        # update energy plots
        for point, energy in zip(energy_points, (kin_energy, pot_energy, tot_energy)):
            point.set_data(times[time_idx], energy[time_idx])

        return bodies + energy_points

    anim = animation(fig, get_frame, frame_count, blit=True, interval=frame_time)

    end = perf_counter()
    print(f"Generating the animation took {end - start:.3f} s")

    if save_to_path:
        start = perf_counter()
        anim.save(save_to_path)
        end = perf_counter()
        print(f"Render took {end - start:.3f} s")
    else:
        plt.show()


def _get_energy_at_time(masses, pos, vel, time_idx):
    """
    Internal function used to calculate kinetic energy and potential energy at
    a give time index using a vectorized direct sum approach. This function is
    necessary to facilitate the parallelization of the energy calculation
    across multiple CPU cores.

    :param masses: Array of masses.
    :param pos: Array of positions over time.
    :param vel: Array of velocities over time.
    :param time_idx: Time index at which the energy is to be calculated.
    :return: Tuple of kinetic energy and potential energy
             at the give time index.
    """

    # kinetic energy
    kin_energy = 0.5 * np.sum(masses * np.sum(vel[:, :, time_idx] ** 2, axis=1))

    # potential energy
    # extract x & y coordinates to a (N, 1) array
    x = pos[:, 0:1, time_idx]
    y = pos[:, 1:2, time_idx]

    # matrices that store pairwise body distances
    dx = x.T - x
    dy = y.T - y

    # calculate pairwise inverse norm of distances
    # mask operation to avoid divide by zero
    norm = np.sqrt(dx ** 2 + dy ** 2)
    inv = np.divide(1, norm, where=norm != 0)

    # multiply matrix element ij with the masses of bodies i and j
    energy_per_body = np.transpose(inv * masses) * masses

    # sum energies
    pot_energy = -0.5 * energy_per_body.sum()

    return kin_energy, pot_energy


def calculate_energy(masses, pos, vel):
    """
    Calculates kinetic energy, gravitational potential energy, and
    total energy for the whole system for each time step in the simulation.

    :param masses: Array of masses.
    :param pos: Array of positions over time.
    :param vel: Array of velocities over time.
    :return: Tuple of kinetic energy over time, potential
             energy over time, and total energy over time.
    """

    with Pool() as pool:
        # distribute calculations across all CPU threads
        result = pool.map(
            partial(_get_energy_at_time, masses, pos, vel),
            range(pos.shape[2]),
        )

    # convert result to NumPy array and extract kinetic and potential energy
    result = np.array(result)
    kin_energy = result[:, 0]
    pot_energy = result[:, 1]

    return kin_energy, pot_energy, kin_energy + pot_energy
