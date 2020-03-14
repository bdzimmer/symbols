"""
Tests for text rendering functionality.
"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import unittest

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

from symbols import text
from symbols import blimp  # TODO: move load_font into it's own location

DEBUG = True
SCRATCH_DIRNAME = "text_scratch"


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

        if DEBUG:
            os.makedirs(SCRATCH_DIRNAME, exist_ok=True)
            offsets = [text.offset(x, font) for x in lines]
            text.animate(SCRATCH_DIRNAME, lines[0:2], offsets, font, width_max, im_func, 1, 0)

    def test_alignment(self):
        """figure out what's going on with text alignment"""

        # 2020-03-13: I thought I had figured out how alignment
        # and offsets work, but I'm still seeing some popping in the
        # multiline stuff. Need to do some tests to precisely
        # understand how all of the measurements work. Don't want
        # to be wondering if line height is correct, etc.

        font = blimp.load_font("Cinzel-Regular.ttf", 64)

        im_bg = Image.new("RGBA", (800, 400), (0, 0, 0))
        draw = ImageDraw.Draw(im_bg)

        # a couple of tests with the raw text drawing functions,
        # none of my own wrappers (other than for methods)

        text_0 = "April"
        text_1 = "2020"
        texts_all = [text_0, text_1]

        start_x = 50
        start_y = 50

        for idx, text_cur in enumerate(texts_all):
            x_pos = start_x + idx * 200
            draw.text(
                (x_pos, start_y),
                text_cur,
                font=font,
                fill=(255, 255, 255))
            print("size:       ", text.size(text_cur, font))
            print("offset:     ", text.offset(text_cur), font)
            print("line height:", text.font_line_height(font))

        if DEBUG:
            os.makedirs(SCRATCH_DIRNAME, exist_ok=True)
            output_filename = os.path.join(SCRATCH_DIRNAME, "align_0.png")
            cv2.imwrite(output_filename, np.array(im_bg))
