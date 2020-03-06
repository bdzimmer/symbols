"""
Tests for text rendering functionality.
"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import unittest

import numpy as np
from PIL import Image

from symbols import text
from symbols import blimp  # TODO: move load_font into it's own location


class TestsText(unittest.TestCase):
    """tests for text rendering functionality"""

    def test_size(self):
        """test size calculation"""
        font = blimp.load_font("consola.ttf", 16)
        width, height = text.size("Hello, world!", font)
        self.assertEqual(width, 117)
        self.assertEqual(height, 15)

    def test_wrap(self):
        """test wrap calculations"""
        font = blimp.load_font("consola.ttf", 18)
        # font = blimp.load_font("times.ttf", 16)

        message = (
            "Summerfield foresaw none of this. But he was not swept along " +
            "by the zeitgiest of Xanthe, as his detractors diminish him. He " +
            "created the political movement that culminated in the Battle of " +
            "Concord and the end of the War on Mars. Four hundred years later, " +
            "the time is ripe for a new movement. But we cannot trust fate to " +
            "bring us another prophet like him. We all must be " +
            "prepared--individually--to act decisively as he did.")

        width_max = 480
        lines = text.wrap_text(message, font, width_max)
        height = text.font_line_height(font) * len(lines)

        im_bg = Image.new("RGBA", (width_max, height), (0, 0, 0, 255))
        print(im_bg.size)

        def im_func(x):
            """helper"""
            x = text.l_to_rgba(x, (0, 0, 255))
            fg = Image.new("RGBA", im_bg.size, (255, 255, 255, 0))
            fg.paste(Image.fromarray(x))
            x = Image.alpha_composite(im_bg, fg)
            x = np.array(x)
            return x

        if True:
            import os
            output_dirname = "text_scratch"
            os.makedirs(output_dirname, exist_ok=True)

            offsets = [text.offset(x, font) for x in lines]
            text.animate(output_dirname, lines[0:2], offsets, font, width_max, im_func, 1, 0)
