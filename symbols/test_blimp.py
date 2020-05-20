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

