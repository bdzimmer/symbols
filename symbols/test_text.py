"""
Tests for text rendering functionality.
"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import unittest

import os

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
        """Draw examples for debugging text alignment issues."""

        font = blimp.load_font("Cinzel-Regular.ttf", 64)

        im_bg = Image.new("RGBA", (1600, 320), (0, 0, 0))

        # a couple of tests with the raw text drawing functions,
        # none of my own wrappers (other than for methods)

        texts_all = [
            "April",
            "2020",
            "April 2020",
            "april",
            "april 2020",
            "a",
            "ap",
            "apr",
            "april",
            "april ",
            "april 2",
            "april 20",
            "april 202",
            "april 2020"
        ]

        start_x = 50
        start_y = 50
        pos_x = start_x
        pos_y = start_y

        for idx, text_cur in enumerate(texts_all):
            # large image
            size_x, size_y = _draw_text_with_info(im_bg, text_cur, font, pos_x, pos_y)
            pos_x = pos_x + size_x + 50

            # small images
            im = Image.new("RGBA", (480, 320), (0, 0, 0))
            _ = _draw_text_with_info(im, text_cur, font, start_x, start_y)
            _debug_save_image(im, f"align_{idx}.png")

        _debug_save_image(im_bg, "align_all.png")


def _debug_save_image(im: Image, filename: str):
    """save an image"""
    if DEBUG:
        os.makedirs(SCRATCH_DIRNAME, exist_ok=True)
        output_filename = os.path.join(SCRATCH_DIRNAME, filename)
        im.save(output_filename)


def _draw_text_with_info(im, text_cur, font, pos_x, pos_y):
    """draw text with info lines for debugging"""

    # ~~~

    # TODO: eventually abstract this stuff out for testing other text libs

    draw = ImageDraw.Draw(im)

    draw.text(
        (pos_x, pos_y),
        text_cur,
        font=font,
        fill=(255, 255, 255))

    size_x, size_y = text.size(text_cur, font)
    offset_x, offset_y = text.offset(text_cur, font)
    line_height = text.font_line_height(font)
    ascent, descent = font.getmetrics()

    # ~~~

    # draw various horizontal lines

    def draw_line(points, color):
        """draw line with offset of pos_x, pos_y"""
        draw.line(
            xy=[(pos_x + x, pos_y + y) for x, y in points],
            fill=color,
            width=1)

    # red at y=0 and y=line_height
    draw_line(
        ((0, 0), (size_x, 0)),
        "red")

    draw_line(
        ((0, line_height), (size_x, line_height)),
        "red")

    # green at offset
    draw_line(
        ((offset_x, offset_y), (offset_x + size_x, offset_y)),
        "green")

    # blue at base of text
    # This should definitely be constant since it only depends
    # on the font, not the text being drawn.
    draw_line(
        ((0, ascent), (size_x, ascent)),
        "blue")

    print("text:       ", "'" + text_cur + "'")
    print("size:       ", size_x, size_y)
    print("offset:     ", offset_x, offset_y)
    print("line height:", line_height)
    print()

    return size_x, size_y
