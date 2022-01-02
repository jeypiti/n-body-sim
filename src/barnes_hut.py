#!/usr/bin/env python3

"""n-body-sim: src/barnes_hut
Implementation of the Barnes-Hut algorithm for calculating
the acceleration caused by the gravitational force.
"""
import numpy as np

__author__ = "jeypiti"
__copyright__ = "Copyright 2021, jeypiti"
__credits__ = ["jeypiti"]
__license__ = "MIT"


# see acceleration function below
positions = np.empty(1)
masses = np.empty(1)


class Quad:
    """
    Quadrant in space. Can either hold no body or one body and has up to four children.

    The index of its sub-quadrant children follows the scheme:
    - Index 0: Lower left sub-quadrant.
    - Index 1: Lower right sub-quadrant.
    - Index 2: Upper left sub-quadrant.
    - Index 3: Upper right sub-quadrant.
    Or graphically:
      --2--  --3--
      --0--  --1--
    """

    def __init__(self, x, y, size):
        """
        :param x: x coordinate of the lower left corner.
        :param y: y coordinate of the lower left corner.
        :param size: Side length of the quadrant.
        """

        self.x = x
        self.y = y

        # storing half the side length is convenient for calculating the position of sub-quadrants
        self.radius = size / 2

        self.body = None  # index of the body held by this quad
        self.children = [None, None, None, None]  # indexing described above

        # Center of mass and total mass of this quad and all of its children
        self.center_of_mass = np.array((0.0, 0.0))
        self.total_mass = 0

    @property
    def is_internal(self):
        """Check whether the quad is internal."""
        return any(self.children)

    @property
    def size(self):
        """Side length of the quad."""
        return self.radius * 2

    def add_body(self, body_idx):
        """
        Distributes the specified body to the quadrant or its children.

        :param body_idx: Index of body to be added.
        """

        # internal node -> distribute new body to children
        if self.is_internal:
            self.get_sub_quad(body_idx).add_body(body_idx)

            self.center_of_mass = (
                self.total_mass * self.center_of_mass + masses[body_idx] * positions[body_idx]
            ) / (self.total_mass + masses[body_idx])
            self.total_mass += masses[body_idx]

        # unfilled external node -> add to self
        elif self.body is None:
            self.body = body_idx

            self.center_of_mass = positions[body_idx]
            self.total_mass = masses[body_idx]

        # filled external node -> distribute current & new body to children
        else:
            self.get_sub_quad(self.body).add_body(self.body)
            self.body = None

            self.get_sub_quad(body_idx).add_body(body_idx)

            self.center_of_mass = (
                self.total_mass * self.center_of_mass + masses[body_idx] * positions[body_idx]
            ) / (self.total_mass + masses[body_idx])
            self.total_mass += masses[body_idx]

    def get_sub_quad_index(self, body_idx):
        """
        Finds the index of the sub-quadrant in which
        the specified body should be placed.

        :param body_idx: Index of body to be placed.
        :return: Sub-quadrant index.
        """

        pos = positions[body_idx, :]

        # right half of quad
        if pos[0] > self.x + self.radius:

            # upper half of quad
            if pos[1] > self.y + self.radius:
                return 3

            return 1

        # left half of quad
        else:

            # upper half of quad
            if pos[1] > self.y + self.radius:
                return 2

            return 0

    def get_sub_quad(self, body_idx):
        """
        Get the sub-quadrant in which the specified body should be placed.
        The sub-quadrant will be created if it doesn't already exist.

        :param body_idx: Index of the body to be placed.
        :return: Sub-quadrant.
        """

        sub_quad_idx = self.get_sub_quad_index(body_idx)

        # create new sub-quad if it doesn't exist
        if self.children[sub_quad_idx] is None:
            new_x = self.x + self.radius * sub_quad_idx % 2
            new_y = self.y + self.radius * (1 if sub_quad_idx >= 2 else 0)

            self.children[sub_quad_idx] = Quad(new_x, new_y, self.radius)

        return self.children[sub_quad_idx]


def _calculate_acceleration(quad, body_idx, theta):
    dist = quad.center_of_mass - positions[body_idx]
    norm_sq = dist.dot(dist)

    # quad is sufficiently far away -> approximate
    # equivalent to s/L < theta
    if 4 * quad.radius ** 2 < theta ** 2 * norm_sq:
        return quad.total_mass * dist * norm_sq ** -1.5

    # internal quad too close -> sum acceleration of children
    elif quad.is_internal:
        acc = np.zeros(2)

        for child in quad.children:
            if child is not None:
                acc += _calculate_acceleration(child, body_idx, theta)

        return acc

    # different external quad -> calculate acceleration directly
    elif quad.body != body_idx:
        return masses[quad.body] * dist * norm_sq ** -1.5

    # same external quad -> no acceleration
    else:
        return np.zeros(2)


def acceleration(m, current_pos, theta=0.5):
    """
    Calculates the acceleration for each body in both x & y direction based on
    the gravitational force using the Barnes-Hut algorithm. For this, it builds
    a quadtree that contains all bodies and then calculates the acceleration
    for each body separately based on the quadtree.

    :param m: A (N,) array of masses.
    :param current_pos: A (N, 2) array of current positions of all bodies.
    :param theta: Threshold value used by the Barnes-Hut algorithm to determine
                  if a quad is sufficiently far away or sufficiently close to a
                  reference body. Based on the result, the quad may be
                  approximated as a single body or be traversed to a deeper level.
    :return: A (N, 2) array of accelerations.
    """

    # Workaround to make positions and masses globally available to the Quad class.
    # This is necessary because each Quad instance needs to know the position of
    # bodies it is interacting with. Alternatively, each Quad instance could store
    # a copy of the positions but this would be more computationally expensive.
    global positions, masses
    positions = current_pos
    masses = m

    # build quadtree
    maxs = current_pos.max(axis=0)  # max values in x & y direction
    mins = current_pos.min(axis=0)  # min values in x & y direction
    root_size = np.max(maxs - mins)
    root_pos = 0.5 * (maxs + mins - root_size)
    root = Quad(root_pos[0], root_pos[1], root_size)

    # fill quadtree
    for body_idx in range(len(m)):
        root.add_body(body_idx)

    # calculate acceleration for all bodies
    result = np.zeros_like(current_pos)
    for body_idx in range(len(m)):
        result[body_idx] = _calculate_acceleration(root, body_idx, theta)

    return result
