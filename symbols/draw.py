"""

Draw geometric shapes.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import cv2

from symbols import symbols
from symbols.symbols import to_int


def draw_line_cv(im, line):
    """draw a line on an image using OpenCV"""
    cv2.line(
        im,
        to_int(line.start),
        to_int(line.end),
        line.color,
        line.thickness,
        cv2.LINE_AA)


def draw_circle_cv(im, circle, frac):
    """draw a circle on an image using OpenCV"""

    end_angle = frac * (circle.end_angle - circle.start_angle) * 360.0 / symbols.TAU
    cv2.ellipse(
        im,
        to_int(circle.center),
        to_int((circle.radius, circle.radius)),
        circle.start_angle * 360.0 / symbols.TAU,
        0.0,
        end_angle,
        circle.color,
        circle.thickness)
