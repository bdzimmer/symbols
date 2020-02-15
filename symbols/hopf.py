"""

Hopf fibration functions.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import numpy as np


def total_points(base_point, fiber_points):
    """find points in the total space S3 from a point in S2 and points S1"""

    a, b, c = base_point

    cos_theta = fiber_points[0, :]
    sin_theta = fiber_points[1, :]

    res = 1.0 / np.sqrt(2.0 * (1.0 + c)) * np.row_stack([
        (1.0 + c) * cos_theta,
        a * sin_theta - b * cos_theta,
        a * cos_theta + b * sin_theta,
        (1.0 + c) * sin_theta
    ])

    return res


def stereographic(points_4d):
    """project 4D points to 3D via the stereographic projection"""
    w = points_4d[0, :]
    points_3d = np.row_stack([
        points_4d[1, :] / (1.0 - w),
        points_4d[2, :] / (1.0 - w),
        points_4d[3, :] / (1.0 - w),
    ])
    return points_3d
