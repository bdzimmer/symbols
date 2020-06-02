"""

Utilities for drawing and animating text.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import re
import os
from typing import List, Any, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw


def size(text, font):
    """get the width and height (in pixels) of a string"""
    return font.getsize(text)


def font_line_height(font):
    """get line height"""
    ascent, descent = font.getmetrics()
    return ascent + descent


def wrap_text(text, font, width_max):
    """split text into lines / words given a width in pixels"""

    # split into words

    words = re.split("\\s+", text)
    lines = []
    start_idx = 0

    for idx in range(1, len(words)):
        line = " ".join(words[start_idx:(idx + 1)])
        width, _ = size(line, font)

        if width > width_max:
            line_new = " ".join(words[start_idx:idx])
            lines.append(line_new)
            start_idx = idx

    line_new = " ".join(words[start_idx:])
    lines.append(line_new)

    return lines


def multiline(lines, font, line_height, image_width, image_height):
    """draw multiline text using PIL ImageDraw"""

    image = Image.new("L", (image_width, image_height), 0)
    draw = ImageDraw.Draw(image)

    for idx, line, in enumerate(lines):
        print(line)
        pos_x = 0
        pos_y = idx * line_height

        draw.text((pos_x, pos_y), line, font=font, fill="white")

    return np.array(image)


def l_to_rgba(im_l, color):
    """create a colorized transparent image from black and white"""

    # TODO: I believe this code is duplicated elsewhere

    totally_transparent_color = (0, 0, 0)

    height, width = im_l.shape

    solid_text = Image.new("RGBA", (width, height), color)
    solid_text_np = np.array(solid_text)
    solid_text_np[:, :, 3] = solid_text_np[:, :, 3] * (im_l / 255.0)
    # set all completely transparent pixels to (something, 0)
    solid_text_np[solid_text_np[:, :, 3] == 0, 0:3] = totally_transparent_color

    return solid_text_np


def animate_characters(
        output_dirname: str,
        lines: List[str],
        font: Any,
        width_max: int,
        im_func: Callable,     # function to update the image before writing to disk
        dup: int,              # frames per character
        dup_end: int           # duplicate frames at end
        ):
    """animate text by individual characters"""

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    line_lengths = [len(x) for x in lines]
    length_all = sum(line_lengths)
    line_height = font_line_height(font)
    image_height = line_height * len(lines)

    idx_frame_out = 0

    for idx_frame in range(length_all):

        print(idx_frame + 1, "/", length_all)

        # calculate lines_mod
        # TODO: other animation modes
        # TODO: optionally include extra "characters" for line ends
        lines_mod = []
        chars_total = 0
        for idx, line in enumerate(lines):
            if idx_frame >= chars_total + line_lengths[idx]:
                lines_mod.append(line)
                chars_total = chars_total + len(line)
            else:
                partial = line[0:(idx_frame - chars_total + 1)]
                lines_mod.append(partial)
                print(partial)
                break

        # TODO: test a case where one word is too long

        img = multiline(
            lines_mod, font, line_height, width_max, image_height)

        for _ in range(dup):
            im_mod = im_func(img)
            cv2.imwrite(
                os.path.join(output_dirname, str(idx_frame_out).rjust(5, "0") + ".png"),
                im_mod)
            idx_frame_out = idx_frame_out + 1

    for _ in range(dup_end):
        im_mod = im_func(img)
        cv2.imwrite(
            os.path.join(output_dirname, str(idx_frame_out).rjust(5, "0") + ".png"),
            im_mod)
        idx_frame_out = idx_frame_out + 1
