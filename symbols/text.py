"""

Utilities for drawing and animating text.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import re
from typing import List, Any, Callable, Tuple

import numpy as np
from PIL import Image, ImageFont

from symbols import blimp_text


def wrap_text(
        text: str,
        font: ImageFont.FreeTypeFont,
        width_max: int) -> List[str]:
    """split text into lines / words given a width in pixels"""

    # split into words
    words = re.split("\\s+", text)
    lines = []
    start_idx = 0

    for idx in range(1, len(words)):
        line = " ".join(words[start_idx:(idx + 1)])
        width, _ = blimp_text.getsize(font, line) # font.getsize(line)

        if width > width_max:
            line_new = " ".join(words[start_idx:idx])
            lines.append(line_new)
            start_idx = idx

    line_new = " ".join(words[start_idx:])
    lines.append(line_new)

    return lines


def multiline(
        lines: List[str],
        font: ImageFont.FreeTypeFont,
        color: Tuple,
        line_height: int,
        image_width: int,
        image_height: int) -> np.ndarray:
    """draw multiline text using PIL ImageDraw"""

    # pylint: disable=too-many-arguments

    # original behavior
    # image = Image.new("L", (image_width, image_height), 0)
    # draw = ImageDraw.Draw(image)
    #     draw.text((pos_x, pos_y), line, font=font, fill="white")

    image = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))

    for idx, line, in enumerate(lines):
        print(line)
        pos_x = 0
        pos_y = idx * line_height

        # TODO: refactor to handle images with borders
        blimp_text.text(image, (pos_x, pos_y), line, font, color, 0)

    return np.array(image)


def animate_characters(
        lines: List[str],
        font: Any,
        color: Tuple,
        width_max: int,
        im_func: Callable,     # function to update the image before writing to disk
        frame_func: Callable,  # function to write frame to disk
        dup: int,              # frames per character
        dup_end: int           # duplicate frames at end
        ) -> None:
    """animate text by individual characters"""

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    line_lengths = [len(x) for x in lines]
    length_all = sum(line_lengths)
    # line_height = font_line_height(font)
    ascent, descent = blimp_text.getmetrics(font)
    line_height = ascent + descent
    # TODO: potentially add leading to line_height
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
            lines_mod, font, color, line_height, width_max, image_height)

        for _ in range(dup):
            im_mod = im_func(img)
            frame_func(im_mod)
            idx_frame_out = idx_frame_out + 1

    for _ in range(dup_end):
        im_mod = im_func(img)
        frame_func(im_mod)
        idx_frame_out = idx_frame_out + 1
