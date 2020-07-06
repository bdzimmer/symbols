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

    # TODO: it's still a bit unclear whether I should use arc or arc_negative
    # to match opencv behavior
    if circle.start_angle > circle.end_angle:
        context.arc_negative(
            circle.center[0], circle.center[1], circle.radius,
            circle.start_angle, circle.end_angle)
    else:
        context.arc(
            circle.center[0], circle.center[1], circle.radius,
            circle.start_angle, circle.end_angle)

    context.stroke()


def draw_polyline(context: cairo.Context, polyline: symbols.Polyline) -> None:
    """draw a polyline"""

    if not polyline.lines:
        return

    # The gotchas for animation and joints will be handled by polyline_frac, not this.

    context.set_line_width(polyline.thickness)
    _set_color(context, polyline.color)
    # context.set_line_join()   # TODO: implement joint type

    context.move_to(polyline.lines[0].start[0], polyline.lines[0].start[1])

    if polyline.closed:
        for line in polyline.lines[:-1]:
            context.line_to(line.end[0], line.end[1])
        context.close_path()
    else:
        for line in polyline.lines:
            context.line_to(line.end[0], line.end[1])

    context.stroke()


def render(canvas: np.ndarray, primitives: List[symbols.Primitive]) -> None:
    """Render using pycairo"""

    height, width, _ = canvas.shape
    surface, context = create_surface(width, height)

    for prim in primitives:
        print("\trender_cv:", prim.__class__.__name__)
        draw(context, prim)

    surface_image = surface_to_image(surface)

    # composite with canvas
    composite_pil = Image.alpha_composite(
        Image.fromarray(canvas),
        surface_image
    )

    # insert into canvas, mutating canvas
    canvas[:, :, :] = np.array(composite_pil)


def draw(context: cairo.Context, prim: symbols.Primitive) -> None:
    """dispatch"""
    if isinstance(prim, symbols.Line):
        draw_line(context, prim)
    elif isinstance(prim, symbols.Circle):
        draw_circle(context, prim)
    elif isinstance(prim, symbols.Polyline):
        draw_polyline(context, prim)


def create_surface(width: int, height: int) -> Tuple[cairo.ImageSurface, cairo.Context]:
    """create surface"""

    # create transparent Cairo canvas
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    context = cairo.Context(surface)
    return surface, context


def surface_to_image(surface: cairo.ImageSurface) -> Image.Image:
    """convert surface to RGBA image"""

    # convert from premultiplied alpha to not premultiplied alpha
    return Image.fromarray(_surface_to_array(surface), mode="RGBa").convert("RGBA")


def _surface_to_array(surface: cairo.ImageSurface) -> np.ndarray:
    """convert surface to array"""

    # NOTE! The format of the array is PREMULTIPLIED ALPHA, not RGBA!
    res = np.ndarray(
        (surface.get_height(), surface.get_width(), 4),  # do height and width work here???
        dtype=np.uint8,
        buffer=surface.get_data())
    res = res[:, :, [2, 1, 0, 3]]  # swap RGB order like OpenCV
    return res


def _set_color(context: cairo.Context, color: Tuple) -> None:
    """set the context color"""

    if len(color) == 4:
        red, green, blue, alpha = color
        context.set_source_rgba(red / 255.0, green / 255.0, blue / 255.0, alpha / 255.0)
    else:
        red, green, blue = color
        context.set_source_rgb(red / 255.0, green / 255.0, blue / 255.0)
