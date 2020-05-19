"""

Tests for drawing geometric shapes.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import numpy as np

from symbols import draw, symbols


def test_draw_line_cv():
    """test drawing a line"""

    im_size = (128, 128, 3)
    im_test = np.zeros(im_size)
    line = symbols.Line((4, 4), (60, 60), (255, 0, 0), 2)

    draw.draw_line_cv(im_test, line)

    assert np.sum(im_test[:, :, 0]) > 0
    assert np.sum(im_test[:, :, 1]) == 0.0
    assert np.sum(im_test[:, :, 2]) == 0.0


def test_draw_circle():
    """test drawing a circle"""

    im_size = (128, 128, 3)
    im_test = np.zeros(im_size)
    circle = symbols.Circle(
        (32, 32), 16, 0.0, symbols.TAU * 0.75, (255, 0, 0), 2)

    draw.draw_circle_cv(im_test, circle, 0.5)

    assert np.sum(im_test[:, :, 0]) > 0
    assert np.sum(im_test[:, :, 1]) == 0.0
    assert np.sum(im_test[:, :, 2]) == 0.0
