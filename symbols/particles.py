"""

Particle emitters and related functionality.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

# ~~~~ particle emitters ~~~~

import random
from typing import List, Tuple, Callable

import cv2
import numpy as np

from symbols import symbols, trim


# type aliases
Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


def build_vertex_emitter(lines: List[symbols.Line]) -> Callable[[], Point2D]:
    """given a set of lines, choose one of the vertices"""
    def emit():
        """emit a particle"""
        line = random.choice(lines)
        if random.random() > 0.5:
            return line.start
        else:
            return line.end
    return emit


def build_line_emitter(lines: List[symbols.Line], weights) -> Callable[[], Point2D]:
    """given a set of lines, choose a point on one of the lines"""

    # build probabilties based on line length
    lengths = [symbols.length(x) for x in lines]
    if weights is not None:
        lengths = [x * y for x, y in zip(lengths, weights)]
    lengths_total = sum(lengths)
    probs = [x / lengths_total for x in lengths]

    idxs = list(range(len(lines)))

    def emit():
        """emit a particle"""
        # choose a line
        line_idx = np.random.choice(idxs, p=probs)
        line = lines[line_idx]
        # interpolate along line
        return symbols.interp(line.start, line.end, random.random())

    return emit


def default_glow_texture(color: Tuple, line_width: int, blur_size: int, glow_strength: float) -> np.ndarray:
    """create a gaussian-blurred point to be later composited"""

    # original blur_size was 31
    # original glow_strength = 2.0

    # determine size necessary and location
    half_width = line_width + blur_size // 2
    width = half_width * 2 + 1  # add another 2??
    texture = np.zeros((width, width, 4), dtype=np.uint8)

    pt_0 = (half_width, half_width)  # add 1?
    pt_1 = pt_0

    if blur_size > 0:
        # set up for alpha blending
        # Fill entire region that will be blurred using a fat line
        # with solid RGB values and alpha 0.
        # The blur will then blur the alpha of the color across the region.
        color_mod = (color[0], color[1], color[2], 0)
        cv2.line(
            texture,
            pt_0,
            pt_1,
            color_mod,
            line_width + blur_size // 2,
            # cv2.LINE_AA  # antialising probably pollutes colors
        )

    cv2.line(
        texture,
        pt_0,
        pt_1,
        color,
        line_width,
        # cv2.LINE_AA  # antialising probably pollutes colors
    )

    if blur_size > 0:
        blurred = cv2.GaussianBlur(
            texture, (blur_size, blur_size), 0)
        # blurred = src_chunk

        if blurred is not None:
            # use only the blur
            # texture = blurred

            # ensure that all pixels of nonzero alpha have the RGB values of color
            # this could be done instead of the fat line above
            # blurred[:, :, 0:3][blurred[:, :, 3] > 0] = (color[0], color[1], color[2])

            # increase intensity of blur and add to original, then clip
            texture = np.clip(blurred * glow_strength + texture, 0.0, 255.0)

            # max blur with original
            # texture = np.maximum(blurred, src_chunk)

    return texture


def draw_texture(
        canvas: np.ndarray,
        pt_xy: Tuple[int, int],
        texture: np.ndarray,
        comp_func: Callable):
    """draw a particle texture onto a canvas with a compositing function"""

    canvas_dim = (canvas.shape[1], canvas.shape[0])
    texture, (pt_x, pt_y) = trim.trim(texture, pt_xy, canvas_dim)

    # print(
    #     pt_xy,
    #     (pt_x, pt_y),
    #     texture.shape, "->",
    #     pt_y, "-", (pt_y + texture.shape[0]), ",",
    #     pt_x, "-", (pt_x + texture.shape[1]))

    canvas_chunk = canvas[pt_y:(pt_y + texture.shape[0]), pt_x:(pt_x + texture.shape[1]), :]
    canvas[pt_y:(pt_y + texture.shape[0]), pt_x:(pt_x + texture.shape[1]), :] = comp_func(texture, canvas_chunk)
