"""

2D drawing functions using pycairo.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from typing import List, Tuple

import numpy as np
import cairo
from PIL import Image

from symbols import symbols


def draw_line(context: cairo.Context, line: symbols.Line) -> None:
    """draw a line"""
    context.set_line_width(line.thickness)
    _set_color(context, line.color)

    context.move_to(line.start[0], line.start[1])
    context.line_to(line.end[0], line.end[1])
    context.stroke()


def draw_circle(context: cairo.Context, circle: symbols.Circle) -> None:
    """draw a circle"""
    context.set_line_width(circle.thickness)
    _set_color(context, circle.color)

    context.arc(
        circle.center[0], circle.center[1], circle.radius,
        circle.start_angle, circle.end_angle)
    context.stroke()


def render(canvas: np.ndarray, primitives: List[symbols.Primitive]) -> None:
    """Render using pycairo"""

    # create transparent Cairo canvas
    height, width, _ = canvas.shape
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)

    # create context from surface
    context = cairo.Context(surface)

    for prim in primitives:
        print("\trender_cv:", prim.__class__.__name__)

        if isinstance(prim, symbols.Line):
            draw_line(context, prim)
        elif isinstance(prim, symbols.Circle):
            draw_circle(context, prim)

    # get array
    surface_array = np.ndarray((height, width, 4), dtype=np.uint8, buffer=surface.get_data())
    # canvas[:, :, :] = surface_array
    # return

    # composite with canvas
    composite_pil = Image.alpha_composite(
        Image.fromarray(canvas),
        Image.fromarray(surface_array)
    )

    # insert into canvas, mutating canvas
    canvas[:, :, :] = np.array(composite_pil)


def _set_color(context: cairo.Context, color: Tuple) -> None:
    """set the context color"""

    if len(color) == 4:
        red, green, blue, alpha = color
        context.set_source_rgba(red / 255.0, green / 255.0, blue / 255.0, alpha / 255.0)
    else:
        red, green, blue = color
        context.set_source_rgb(red / 255.0, green / 255.0, blue / 255.0)
