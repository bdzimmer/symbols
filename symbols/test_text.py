"""
Tests for text rendering functionality.
"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import os
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from symbols import blimp, text, text_scala, util

DEBUG = True
SCRATCH_DIRNAME = os.path.join("test_scratch", "text")


def test_size():
    """test size calculation"""
    font = blimp.load_font("consola.ttf", 16)
    width, height = text.size("Hello, world!", font)
    assert width == 117
    assert height == 15


def test_wrap():
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

    def im_func(img):
        """helper"""
        img = text.l_to_rgba(img, (0, 0, 255))
        im_fg = Image.new("RGBA", im_bg.size, (0, 0, 0, 0))
        im_fg.paste(Image.fromarray(img))
        img = Image.alpha_composite(im_bg, im_fg)
        img = np.array(img)
        return img

    if DEBUG:
        scratch_dirname = os.path.join(SCRATCH_DIRNAME, "wrap")
        os.makedirs(scratch_dirname, exist_ok=True)
        frame_func = _build_frame_writer(scratch_dirname)
    else:
        frame_func = lambda x: None

    text.animate_characters(lines[0:2], font, width_max, im_func, frame_func, 1, 3)

    if DEBUG:
        movie_filename = os.path.join(SCRATCH_DIRNAME, "wrap.mp4")
        fps = 30
        ffmpeg_command = util.ffmpeg_command(
            scratch_dirname, movie_filename, width_max, height, fps)
        os.system(ffmpeg_command)


def test_alignment():
    """Draw examples for debugging text alignment issues."""
    # pylint: disable=too-many-locals

    font = blimp.load_font("Cinzel-Regular.ttf", 48)
    font_scala = ("Cinzel", "plain", 48)
    border_size = (32, 32)

    # 2020-03-22

    # These samples demonstrate a one pixel shift off the baseline
    # using Cinzel Regular size 48 when using PIL's simple font
    # drawing functionality

    # The shift happens on the first zero of 2020, and all of the
    # glyphs shift upward.

    # The shift does not necessarily happen for other font sizes,
    # such as 49 and 64.

    def draw_pil(img, text_cur, pos_x, pos_y):
        """draw text with PIL"""
        draw = ImageDraw.Draw(img)
        draw.text(
            (pos_x, pos_y),
            text_cur,
            font=font,
            fill=(255, 255, 255))
        size_x, size_y = text.size(text_cur, font)
        # offset_x, offset_y = text.offset(text_cur, font)
        line_height = text.font_line_height(font)
        ascent, descent = font.getmetrics()
        return size_x, size_y, ascent, descent, line_height

    def draw_scala(img, text_cur, pos_x, pos_y):
        """draw text with scala"""
        info = text_scala.draw_on_image(
            img,
            (pos_x, pos_y),
            text_cur,
            font=font_scala,
            fill=(255, 255, 255),
            border_size=border_size,
            stroke_width=0)
        size_x = info["width"]
        size_y = info["height"]
        ascent = info["ascent"]
        descent = info["descent"]
        line_height = ascent + descent
        return size_x, size_y, ascent, descent, line_height

    texts_all = [
        "April",
        "2020",
        "April 2020",
        "april",
        "april 2020",
        "a",
        "ap",
        "apr",
        "apri",
        "april",
        "april ",
        "april 2",
        "april 20",
        "april 202",
        "april 2020"
    ]

    # some examples that are nice for testing at kerning:
    # texts_all = ["AVIARY", "aviary", "To"]

    start_x = 50
    start_y = 50

    for draw_method in ["pil", "scala"]:
        im_bg = Image.new("RGBA", (2000, 320), (0, 0, 0))

        draw_func = draw_scala if draw_method == "scala" else draw_pil
        pos_x = start_x
        pos_y = start_y

        for idx, text_cur in enumerate(texts_all):
            # large image
            size_x, _ = _draw_text_with_info(
                im_bg, text_cur, pos_x, pos_y, draw_func)
            pos_x = pos_x + size_x + 50

            # small images
            img = Image.new("RGBA", (480, 320), (0, 0, 0))
            _ = _draw_text_with_info(
                img, text_cur, start_x, start_y, draw_func)
            _debug_save_image(img, "align", f"align_{idx}_{draw_method}.png")

        _debug_save_image(im_bg, ".", f"align_all_{draw_method}.png")


def _debug_save_image(img: Image, subdirname: str, filename: str):
    """save an image"""
    if DEBUG:
        scratch_dirname = os.path.join(SCRATCH_DIRNAME, subdirname)
        os.makedirs(scratch_dirname, exist_ok=True)
        output_filename = os.path.join(scratch_dirname, filename)
        img.save(output_filename)


def _draw_text_with_info(
        img: Image, text_cur: str, pos_x: int, pos_y: int, draw_func: Callable):
    """draw text with info lines for debugging"""

    draw = ImageDraw.Draw(img)

    res = draw_func(img, text_cur, pos_x, pos_y)
    size_x, size_y, ascent, _, line_height = res

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

    # blue at base of text
    # This should definitely be constant since it only depends
    # on the font, not the text being drawn.
    draw_line(
        ((0, ascent), (size_x, ascent)),
        "blue")

    print("text:       ", "'" + text_cur + "'")
    print("size:       ", size_x, size_y)
    print("line height:", line_height)
    print()

    return size_x, size_y


def _build_frame_writer(output_dirname: str) -> Callable:
    """frame writer"""
    idx = [0]

    def write(img: Image.Image):
        """write and increment counter"""
        cv2.imwrite(
            os.path.join(output_dirname, str(idx[0]).rjust(5, "0") + ".png"),
            img[:, :, [2, 1, 0]])
        idx[0] = idx[0] + 1

    return write
