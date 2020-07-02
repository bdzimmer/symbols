"""

Tests for drawing geometric shapes.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import os

import numpy as np

from symbols import draw_cv, draw_cairo, symbols, debugutil

DEBUG_VISUALIZE = os.environ.get("DEBUG_VISUALIZE") == "true"


def test_draw():
    """integration test for drawing"""

    im_size = (256, 256, 3)

    # a couple of notes as of 2020-07-02:

    # I thought I had got the same behavior out of the cairo
    # circle rendering as open cv, but this isn't working the same.

    # The test is very simple:
    # - range of red channel values in opencv vs only {0, 255} in red channel for cairo
    # - only zeros in green and blue channel
    # - range of alpha values.
    # You can easily come up with valid primitves where this is not true.

    line = symbols.Line((16, 16), (240, 240), 2, 0, (255, 0, 0))
    circle = symbols.Circle(
        (128, 128), 64, 0.0, symbols.TAU * 0.75, 2, 0, (255, 0, 0))
    polyline_closed = symbols.Polyline(
        [
            symbols.Line((16, 16), (240, 16), None, None, None),
            symbols.Line((240, 16), (240, 240), None, None, None),
            symbols.Line((240, 240), (16, 16), None, None, None)
        ],
        "", True, 2, 0, (255, 0, 0))
    polyline_open = symbols.Polyline(
        [
            symbols.Line((16, 16), (16, 240), None, None, None),
            symbols.Line((16, 240), (240, 16), None, None, None)
        ],
        "", False, 2, 0, (255, 0, 0))
    polyline_empty = symbols.Polyline([], "", False, 2, 0, (255, 0, 0))

    primitives = [line, circle, polyline_closed, polyline_open]

    for prim in primitives:

        # test opencv drawing

        im_test = np.zeros(im_size, dtype=np.uint8)

        draw_cv.draw(im_test, prim)

        if DEBUG_VISUALIZE:
            debugutil.show(im_test, "test_draw_cv")

        if not isinstance(prim, symbols.Polyline):  # no opencv polyline capability yet
            assert len(set(np.unique(im_test[:, :, 0]))) > 2
            assert np.alltrue(im_test[:, :, 1] == 0)
            assert np.alltrue(im_test[:, :, 2] == 0)

        # test cairo drawing

        surf, con = draw_cairo.create_surface(im_size[0], im_size[1])
        draw_cairo.draw(con, prim)
        im_test = np.array(draw_cairo.surface_to_image(surf))

        if DEBUG_VISUALIZE:
            debugutil.show(im_test, "test_draw_cairo")

        assert set(np.unique(im_test[:, :, 0])) == {0, 255}
        assert np.alltrue(im_test[:, :, 1] == 0)
        assert np.alltrue(im_test[:, :, 2] == 0)
        assert len(set(np.unique(im_test[:, :, 3]))) > 2

    # draw all primitives at once

    im_test = np.zeros(im_size, dtype=np.uint8)
    draw_cv.render(im_test, primitives + [polyline_empty])

    if DEBUG_VISUALIZE:
        debugutil.show(im_test, "test_draw_cv")

    assert len(set(np.unique(im_test[:, :, 0]))) > 2
    assert np.alltrue(im_test[:, :, 1] == 0)
    assert np.alltrue(im_test[:, :, 2] == 0)

    im_test = np.zeros((im_size[0], im_size[1], 4), dtype=np.uint8)
    draw_cairo.render(im_test, primitives + [polyline_empty])

    if DEBUG_VISUALIZE:
        debugutil.show(im_test, "test_draw_cairo")

    assert set(np.unique(im_test[:, :, 0])) == {0, 255}
    assert np.alltrue(im_test[:, :, 1] == 0)
    assert np.alltrue(im_test[:, :, 2] == 0)
    assert len(set(np.unique(im_test[:, :, 3]))) > 2
