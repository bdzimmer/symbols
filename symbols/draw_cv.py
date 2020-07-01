"""

Draw geometric shapes.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from typing import List

import cv2
import numpy as np

from symbols import symbols
from symbols.symbols import to_int


def draw_line(img, line) -> None:
    """draw a line on an image using OpenCV"""
    cv2.line(
        img,
        to_int(line.start),
        to_int(line.end),
        (line.color[2], line.color[1], line.color[0]),
        line.thickness,
        cv2.LINE_AA)


def draw_circle(img, circle) -> None:
    """draw a circle on an image using OpenCV"""

    # end_angle = frac * (circle.end_angle - circle.start_angle) * 360.0 / symbols.TAU

    cv2.ellipse(
        img,
        to_int(circle.center),
        to_int((circle.radius, circle.radius)),
        circle.start_angle * 360.0 / symbols.TAU,
        0.0,
        # end_angle,
        (circle.end_angle - circle.start_angle) * 360.0 / symbols.TAU,
        (circle.color[2], circle.color[1], circle.color[0]),
        circle.thickness)


# TODO: polyline drawing


def render(canvas: np.ndarray, primitives: List[symbols.Primitive]) -> None:
    """Render a sorted list of primitives using OpenCV"""

    for prim in primitives:
        print("\trender_cv:", prim.__class__.__name__)

        if isinstance(prim, symbols.Line):
            draw_line(canvas, prim)
        elif isinstance(prim, symbols.Circle):
            draw_circle(canvas, prim)
