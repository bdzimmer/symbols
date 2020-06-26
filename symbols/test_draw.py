"""

Tests for drawing geometric shapes.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import os

import numpy as np

from symbols import draw, symbols, debugutil

DEBUG_VISUALIZE = os.environ.get("DEBUG_VISUALIZE") == "true"


def test_draw_line_cv():
    """test drawing a line"""

    im_size = (256, 256, 3)
    im_test = np.zeros(im_size)
    line = symbols.Line((16, 16), (240, 240), 2, 0, (255, 0, 0))

    draw.draw_line_cv(im_test, line)

    assert np.sum(im_test[:, :, 0]) > 0
    assert np.sum(im_test[:, :, 1]) == 0.0
    assert np.sum(im_test[:, :, 2]) == 0.0

    if DEBUG_VISUALIZE:
        debugutil.show(im_test, "test_draw_line_cv")


def test_draw_circle_cv():
    """test drawing a circle"""

    im_size = (256, 256, 3)
    im_test = np.zeros(im_size)
    circle = symbols.Circle(
        (128, 128), 64, 0.0, symbols.TAU * 0.75, 2, 0, (255, 0, 0))

    draw.draw_circle_cv(im_test, circle)

    assert np.sum(im_test[:, :, 0]) > 0
    assert np.sum(im_test[:, :, 1]) == 0.0
    assert np.sum(im_test[:, :, 2]) == 0.0

    if DEBUG_VISUALIZE:
        debugutil.show(im_test, "test_draw_circle_cv")
