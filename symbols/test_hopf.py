"""
Unit tests for Hopf fibration and related visualization funcs.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import unittest

import os
import time

import numpy as np
import cv2

from symbols import effects, hopf, transforms, util


class TestsHopf(unittest.TestCase):
    """aldsfjahldkjfh"""

    def test_draw_fibration(self):
        """integration test for drawing hopf links"""

        scratch_dirname = "scratch_hopf"
        os.makedirs(scratch_dirname, exist_ok=True)

        width = 800
        height = 800
        color = (50, 0, 200)  # (230, 230, 0)
        line_width = 2        # 1
        n_points = 128        # number of points in each ring

        # look down at origin from z = 5
        cam_trans = transforms.camera_transform(np.identity(3), np.array([0.0, 0.0, 4.0]))
        view_pos = np.array([0.0, 0.0, 500.0])

        # 3D points in a flat circle in the XY plane

        circle_points = transforms.points_around_circle(n_points, 0.0, 1.0)
        circle_points = np.array(circle_points)
        circle_points = np.column_stack((circle_points, np.zeros(len(circle_points))))
        circle_points = np.transpose(circle_points)

        fiber_points = circle_points

        output_prefix = "hopf_links"
        output_filename = os.path.join(scratch_dirname, output_prefix + ".png")

        base_points = circle_points[:, ::16]

        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # TODO: separate set of sorted lines from drawing sorted lines, lol

        start_time = time.time()

        for _ in range(3):
            hopf.draw_fibration(
                canvas,
                base_points, fiber_points,
                np.identity(4),
                cam_trans, view_pos,
                color, line_width)

        total_time = time.time() - start_time
        print("draw_fibration:", total_time, "sec")

        cv2.imwrite(output_filename, canvas)
