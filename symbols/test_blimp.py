"""

Unit / integration tests for blimp.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import os

from PIL import ImageFont

from symbols import blimp, debugutil


DEBUG_VISUALIZE = os.environ.get("DEBUG_VISUALIZE") == "true"


def test_load_font():
    """test font loading"""
    font = blimp.load_font("consola.ttf", 100)
    assert isinstance(font, ImageFont.FreeTypeFont)


def test_text_standard():
    """test text rendering"""
    # some of this might belong more in a test for blimp_text,
    # but I think that module will go away eventually.

    font = blimp.load_font("consola.ttf", 100)
    args_list = [
        ("test", font, (0, 255, 255), 0, None),
        ("test", font, None, 2, (0, 255, 255)),
        ("test", font, (0, 255, 255, 80), 0, None),
        ("test", font, None, 2, (0, 255, 255, 80))
    ]

    for args in args_list:
        im_res = blimp.text_standard(*args)
        assert im_res.size == (220, 101)
        assert im_res.mode == "RGBA"
        if DEBUG_VISUALIZE:
             debugutil.show(im_res, "test_text_standard")
