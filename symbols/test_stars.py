"""

Tests for stars.

"""

from functools import partial as P
import math
import os

import numpy as np

from symbols import stars, draw_cairo, symbols, debugutil, func, util


DEBUG = True

SCRATCH_DIRNAME = os.path.join("test_scratch", "stars")


def test_star_points():
    """test star_points"""

    image_width = 640
    star_radius = 256
    center = (image_width // 2, image_width // 2)

    points_0 = stars.star_points(7, 3)
    points_1 = stars.star_points(7, 2)

    polyline_0 = symbols.Polyline(
        center=(0, 0),
        lines=[symbols.Line(x, y, None, None, None) for x, y in stars.line_loop(points_0)],
        joint_type="",
        closed=True,
        thickness=8,
        depth=0,
        color=(0, 0, 0))

    polyline_1 = symbols.Polyline(
        center=(0, 0),
        lines=[symbols.Line(x, y, None, None, None) for x, y in stars.line_loop(points_1)],
        joint_type="",
        closed=True,
        thickness=24,
        depth=0,
        color=(0, 0, 0))

    assert len(points_0) == 7
    assert len(points_1) == 7

    if DEBUG:

        os.makedirs(SCRATCH_DIRNAME, exist_ok=True)
        frame_writer = util.build_frame_writer(SCRATCH_DIRNAME, None)

        n_frames = 100
        max_angle = math.pi / 2

        for idx in range(n_frames):

            frac = idx / n_frames

            polyline_0_animated = func.apply(
                polyline_0,
                P(symbols.scale_center_primitive, fac=star_radius, center=(0, 0)),
                P(symbols.rotate_center_primitive, rad=frac * max_angle, center=(0, 0)),
                P(symbols.translate_primitive, trans=center))

            polyline_1_animated = func.apply(
                polyline_1,
                P(symbols.scale_center_primitive, fac=star_radius, center=(0, 0)),
                P(symbols.rotate_center_primitive, rad=frac * max_angle, center=(0, 0)),
                P(symbols.translate_primitive, trans=center))

            surface, context = draw_cairo.create_surface(image_width, image_width)
            draw_cairo.draw_polyline(context, polyline_0_animated)
            draw_cairo.draw_polyline(context, polyline_1_animated)
            star_image = draw_cairo.surface_to_image(surface)
            # filename = "star.png"
            # debugutil.save_image(star_image, SCRATCH_DIRNAME, filename)

            frame_writer(np.array(star_image))
