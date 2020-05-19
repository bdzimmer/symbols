"""
Unit tests for Hopf fibration and related visualization funcs.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import os
import time

import cv2
import numpy as np

from symbols import hopf, transforms, symbols


def test_draw_fibration():
    """integration test for drawing hopf links"""
    # pylint: disable=too-many-locals

    scratch_dirname = os.path.join("test_scratch", "hopf")
    output_prefix = "hopf_links"
    os.makedirs(scratch_dirname, exist_ok=True)
    output_filename = os.path.join(scratch_dirname, output_prefix + ".png")

    width = 1280
    height = 720
    color = (50, 0, 230, 255)
    line_width = 2
    n_base_points = 7     # 7 links
    n_fiber_points = 128  # number of points in each ring
    tube_radius = 0.15
    tube_n_points = 16
    decoration_only = False

    # look down at origin from z = 4.0
    cam_trans = transforms.camera_transform(
        np.identity(3), np.array([0.0, 0.0, 4.0]))
    view_pos = np.array([0.0, 0.0, 500.0])

    def get_circle_points(n_points, start_angle):
        """3D points in a flat circle in the XY plane"""
        circle_points = transforms.points_around_circle(n_points, start_angle, 1.0)
        circle_points = np.array(circle_points)
        circle_points = np.column_stack((circle_points, np.zeros(len(circle_points))))
        circle_points = np.transpose(circle_points)
        return circle_points

    fiber_points = get_circle_points(n_fiber_points, 0.0)
    base_points = get_circle_points(n_base_points, 0.0 * symbols.TAU / 7.0)

    def lighting_func(_, norms):
        """create a lighting function"""
        return hopf.lighting_func_generic(norms, color, 2.0, False)

    def decoration_func(x_vec, z_vec):
        """generate additional points for each projected fiber point"""
        x_vec = x_vec - z_vec * np.dot(x_vec, z_vec)
        x_vec = x_vec / np.linalg.norm(x_vec)
        return hopf.disk_segments(
            x_vec, z_vec, tube_radius, tube_n_points)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    start_time = time.time()

    hopf.draw_fibration(
        canvas,
        base_points, fiber_points,
        np.identity(4),
        cam_trans, view_pos,
        lighting_func, line_width,
        decoration_func, decoration_only)

    total_time = time.time() - start_time
    print("draw_fibration:", total_time, "sec")

    cv2.imwrite(output_filename, canvas)
    print("output file:", output_filename)
