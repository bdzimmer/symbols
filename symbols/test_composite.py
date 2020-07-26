"""
Tests for compositing methods.
"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import os
from typing import Tuple

import numpy as np
from PIL import Image

from symbols import composite
from symbols import draw_cairo, symbols, transforms
from symbols import debugutil


DEBUG = True
SCRATCH_DIRNAME = os.path.join("test_scratch", "composite")


def test_blend():
    """test blending functions"""

    canvas_x = 1280
    canvas_y = 1280
    center = (canvas_x // 2, canvas_y // 2)

    blend_funcs = {
        "alpha": composite.alpha_blend,
        "additive": composite.additive_blend
    }

    triangle_points = [
        symbols.add(x, center)
        for x in transforms.points_around_circle(3, symbols.TAU * 0.25, 200)]

    alpha = 128
    dots = [
        _dot_image(800, x)
        for x in [(255, 0, 0, alpha), (0, 255, 0, alpha), (0, 0, 255, alpha)]]

    for blend_method_name in ["alpha", "additive"]:
        canvas = np.zeros((canvas_y, canvas_x, 4), dtype=np.uint8)
        # canvas[:, :, 0:4] = (64, 64, 64, 255)
        blend_func = blend_funcs[blend_method_name]
        for tri_pt, dot in zip(triangle_points, dots):
            c_x, c_y = symbols.add((int(tri_pt[0]), int(tri_pt[1])), (-400, -400))
            res = blend_func(dot, canvas[c_y:(c_y + 800), c_x:(c_x + 800), :])

            print(
                blend_method_name, "\n",
                "\tdot:", np.min(dot[:, :, 3]), np.max(dot[:, :, 3]), "\n",
                "\tdst rgb:", np.min(canvas[c_y:(c_y + 800), c_x:(c_x + 800), 0:3]),
                np.max(canvas[c_y:(c_y + 800), c_x:(c_x + 800), 0:3]), "\n",
                "\tdst a:", np.min(canvas[c_y:(c_y + 800), c_x:(c_x + 800), 3]),
                np.max(canvas[c_y:(c_y + 800), c_x:(c_x + 800), 3]), "\n",
                "\tres a:", np.min(res[:, :, 3]), np.max(res[:, :, 3]))

            canvas[c_y:(c_y + 800), c_x:(c_x + 800), :] = res

        # TODO: assertions

        debugutil.save_image(
            Image.fromarray(canvas), SCRATCH_DIRNAME, f"dots_{blend_method_name}.png")


def _dot_image(diameter: float, color: Tuple) -> np.ndarray:
    """get an image of a dot"""
    surface, context = draw_cairo.create_surface(diameter, diameter)
    draw_cairo.draw_dot(
        context,
        symbols.Dot((diameter / 2, diameter / 2), diameter / 2, 0, 0, color))
    dot_image = draw_cairo.surface_to_image(surface)
    return np.array(dot_image)
