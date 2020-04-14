"""
Calculate and apply transforms.
"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import numpy as np
import cv2

from symbols import symbols


def transformation(rot, trans):
    """construct a rigid transformation"""
    res = np.identity(4)
    res[0:3, 0:3] = rot
    res[0:3, 3] = trans
    return res


def sep_rotation(transf):
    """extract the rotation from a rigid transformation"""
    return transf[0:3, 0:3]


def transform(mat, points):
    """apply a rigid transformation"""
    return np.dot(mat[0:3, 0:3], points) + mat[0:3, 3:4]


def camera_transform(rot, pos):
    """construct a camera transform"""

    translation = transformation(np.identity(3), 0.0 - pos)
    rotation = transformation(rot, np.zeros((3,)))

    return np.dot(rotation, translation)


def perspective(points, view_pos):
    points = np.copy(points)
    points[2, :] = np.minimum(points[2, :], 0.0)

    return np.row_stack([
        view_pos[2] * points[0, :] / points[2, :] - view_pos[0],
        view_pos[2] * points[1, :] / points[2, :] - view_pos[1]])


def points_around_circle(n_points, start, radius):
    return [
        symbols.circle_point(x * 1.0 * symbols.TAU / n_points + start, radius)
        for x in range(n_points)]


def draw_polyline(im, points_p, colors):
    """draw a polyline on an image"""

    point_idxs = list(range(points_p.shape[1])) + [0]

    for idx in range(len(point_idxs) - 1):
        p0 = tuple(points_p[:, point_idxs[idx]])
        p1 = tuple(points_p[:, point_idxs[idx + 1]])
        color = tuple(colors[:, idx])

        # color_strength = (np.sin(2.0 * np.pi * (idx * 1.0) / len(point_idxs)) + 1.0) * 0.5

        # some goofy stuff going on here with color
        # print(idx, tuple(colors[idx]), [type(x) for x in colors[idx]])

        # a proxy for backface
        if color != (0.0, 0.0, 0.0):

            cv2.line(
                im,
                p0,
                p1,
                color,
                2,
                cv2.LINE_AA)
