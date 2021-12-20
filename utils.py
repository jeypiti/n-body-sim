#!/usr/bin/env python3

"""n-body-sim: /utils
File holding various utilities.
"""
from functools import wraps
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation as animation


__author__ = "jeypiti"
__copyright__ = "Copyright 2021, jeypiti"
__credits__ = ["jeypiti"]
__license__ = "MIT"


def solver(func):
    """
    Decorator for solver functions. Handles basic errors
    and sets up data structures for the simulation.

    :param func: Solver function.
    :return: Wrapped solver function.
    """

    @wraps(func)
    def wrapper(masses, init_pos, init_vel, time_steps, dt):

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

        return func(masses, pos, vel, dt)

    return wrapper


def animate(pos, duration=3, max_frame_rate=60, save_to_path=None):
    """
    Produces an animation of a given array of positions.
    Optionally saves the animation to a video file if a path is specified.

    :param pos: Array of positions over time.
    :param duration: Duration of the animation in seconds.
    :param max_frame_rate: Maximum frame rate at which the animation
                           should be displayed or saved.
    :param save_to_path: If given, saves the animation to this file path.
    """

    fig, ax = plt.subplots()

    ax.set_xlim(np.min(pos[:, 0, :]), np.max(pos[:, 0, :]))
    ax.set_ylim(np.min(pos[:, 1, :]), np.max(pos[:, 1, :]))
    # ax.set_ylim(-1, 1)
    # ax.set_aspect("equal")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    # prepare a list of bodies for animation
    points = [ax.plot([], [], marker="o")[0] for _ in range(pos.shape[0])]

    frame_count = int(min(pos.shape[2], duration * max_frame_rate))
    frame_step = pos.shape[2] / frame_count
    frame_rate = frame_count / duration
    frame_time = max(1.0, 1000 / frame_rate)

    print(
        f"Rendering {frame_count * frame_time / 1000:.1f}s animation @ {frame_rate:.1f} FPS"
        f" ({frame_time:.1f} ms per frame)\nShowing {frame_count} frames from a total of"
        f" {pos.shape[2]} time steps (one frame every {frame_step:.1f} steps)"
    )

    def get_frame(frame_idx):
        for mass_idx, point in enumerate(points):
            point.set_data(pos[mass_idx, :, int(frame_idx * frame_step)])

        return points

    anim = animation(fig, get_frame, frame_count, blit=True, interval=frame_time)

    if save_to_path:
        start = perf_counter()
        anim.save(save_to_path)
        end = perf_counter()
        print(f"Render took {end - start:.3f} s")
    else:
        plt.show()
