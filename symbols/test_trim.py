"""

Tests for image trimming functions.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import os

import numpy as np
from PIL import Image

from symbols import trim, debugutil

DEBUG_VISUALIZE = os.environ.get("DEBUG_VISUALIZE") == "true"


def test_trim():
    """test layer trimming, with and without borders"""

    canvas_dim = (128, 128)

    # layer_dim, layer_xy, border_xy, size_expected, xy_expected
    tests = [
        # ~~~ no borders

        # no trimming or adjustment
        ((64, 32), (16, 16), (0, 0), (64, 32), (16, 16)),

        # trim x, adjust x
        ((64, 32), (-16, 16), (0, 0), (64 - 16, 32), (0, 16)),

        # trim x and y, adjust x and y
        ((64, 32), (-16, -8), (0, 0), (64 - 16, 32 - 8), (0, 0)),

        # trim x, no adjustment
        ((64, 32), (80, 16), (0, 0), (64 - 16, 32), (80, 16)),

        # trim x and y, no adjustment
        ((64, 32), (80, 104), (0, 0), (64 - 16, 32 - 8), (80, 104)),

        # ~~~~ with borders

        # no trimming or adjustment
        ((64, 32), (16, 16), (4, 8), (64 + 4 * 2, 32 + 8 * 2), (16, 16)),

        # trim x and y, adjust x and y
        # left hand and upper borders lie off edge of canvas
        # trim only removes left hand and upper borders
        # adjustment for border size subtraction
        ((64, 32), (0, 0), (4, 8), (64 + 4 * 2 - 4, 32 + 8 * 2 - 8), (4, 8)),

        # trim x, adjust x
        # x: 16px as well as left hand border are wiped out
        # layer x is adjusted over 4 pixels for future subtraction of border size
        ((64, 32), (-16, 16), (4, 8), (64 + 4 * 2 - 16 - 4, 32 + 8 * 2), (4, 16)),

        # trim x and y, adjust x and y
        # x: 16px and left hand border are wiped out
        # y: 8px and upper border are wiped out
        # layer x and y are both adjusted
        ((64, 32), (-16, -8), (4, 8), (64 + 4 * 2 - 16 - 4, 32 + 8 * 2 - 8 - 8), (4, 8)),

        # trim x and y, no adjustment
        # right hand and lower borders lie off edge of canvas
        # trim only removes right hand and lower borders
        ((64, 32), (64, 96), (4, 8), (64 + 4 * 2 - 4, 32 + 8 * 2 - 8), (64, 96)),

        # trim x, no adjustment
        # x: 16px as well as right hand border are wiped out
        ((64, 32), (80, 16), (4, 8), (64 + 4 * 2 - 16 - 4, 32 + 8 * 2), (80, 16)),

        # trim x and y, no adjustment
        # x: 16px as well as right hand border are wiped out
        # y: 8px as well as bottom border are wiped out
        ((64, 32), (80, 104), (4, 8), (64 + 4 * 2 - 16 - 4, 32 + 8 * 2 - 8 - 8), (80, 104))
    ]

    for layer_dim, layer_xy, border_xy, size_expected, xy_expected in tests:
        layer_im = np.zeros(
            (layer_dim[1] + border_xy[1] * 2, layer_dim[0] + border_xy[0] * 2, 4),
            dtype=np.ubyte)
        layer_im[:, :, 0:4] = (255, 0, 0, 255)  # red borders
        layer_im[
            border_xy[1]:(border_xy[1] + layer_dim[1]),
            border_xy[0]:(border_xy[0] + layer_dim[0]), 0:3] = (0, 255, 255)  # blue content edge
        layer_im[
            (border_xy[1] + 1):(border_xy[1] + layer_dim[1] - 1),
            (border_xy[0] + 1):(border_xy[0] + layer_dim[0] - 1), 0:3] = (0, 255, 0)  # green content

        layer_trimmed, xy_actual = trim.trim_border(
            layer_im, layer_xy, border_xy, canvas_dim)
        size_actual = (layer_trimmed.shape[1], layer_trimmed.shape[0])
        assert size_expected == size_actual
        assert xy_expected == xy_actual

        if DEBUG_VISUALIZE:
            canvas = np.zeros((canvas_dim[1], canvas_dim[0], 4), dtype=np.ubyte)
            canvas[:, :, 0:4] = (128, 128, 128, 255)
            canvas_pil = Image.fromarray(canvas)
            layer_pil = Image.fromarray(layer_trimmed)
            canvas_pil.alpha_composite(
                layer_pil,
                (xy_actual[0] - border_xy[0], xy_actual[1] - border_xy[1]))
            print(layer_dim, layer_xy, border_xy, size_expected, xy_expected)
            debugutil.show(canvas_pil, "composite")
