#!/usr/bin/env python3

"""n-body-sim: benchmark/bodies_acceleration
Benchmark to measure the performance of the different methods of calculating
the acceleration relative to the number of bodies in the simulation.
"""
import gc
import sys
from itertools import product, repeat
from multiprocessing import Pool
from os.path import realpath
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np

sys.path.append(realpath(__file__ + "/../../src"))  # ensure that project files can be imported

import barnes_hut
import direct_sum
from init_cond import generate_planetary_system


__author__ = "jeypiti"
__copyright__ = "Copyright 2022, jeypiti"
__credits__ = ["jeypiti"]
__license__ = "MIT"


seed = 1234567890


def time_simulation(acc_func, num_bodies, number=10):
    assert isinstance(number, int), "number must be an integer"

    masses, pos, _ = generate_planetary_system(num_bodies, seed=seed)

    gc_state = gc.isenabled()
    gc.disable()

    try:
        start = perf_counter()
        for _ in repeat(None, number):
            acc_func(masses, pos)
        end = perf_counter()
        total_time = end - start
    finally:
        if gc_state:
            gc.enable()

    return total_time / number


def test_acc(bodies_upper, steps=10):
    bodies = np.linspace(2, bodies_upper, num=steps, dtype=int)
    acc_funcs = (direct_sum.acceleration, direct_sum.acceleration_vec, barnes_hut.acceleration)

    with Pool() as pool:
        result = pool.starmap_async(
            time_simulation,
            product(acc_funcs, bodies),
        ).get()

    for i, func in enumerate(("Direct sum", "Vec. direct sum", "Barnes-Hut")):
        times = result[len(bodies) * i : len(bodies) * (i + 1)]
        plt.scatter(bodies, times, label=func)

    plt.xlabel(r"$n$")
    plt.ylabel(r"$t\,/\,\mathrm{s}$")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    test_acc(100, steps=5)
