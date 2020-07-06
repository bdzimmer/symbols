"""

Stuff for refactoring text in blimp.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from typing import Any, Tuple

from PIL import ImageDraw
from PIL.ImageFont import FreeTypeFont

from symbols import text_scala


USE_PIL = False
BORDER_DEFAULT = (16, 16)


def text(
        image,
        xy: Tuple[int, int],
        text_str: str,
        font: Any,
        fill: text_scala.Color,
        stroke_width: int) -> None:

    """Draw text on an image, mutating it."""

    # pylint: disable=too-many-arguments

    # Draw text with either fill or stroke.
    # (text_scala does not support both at the same time)

    # TODO: make BORDER_DEFAULT a parameter

    if USE_PIL:
        # TODO: this may have to be revised now, lol
        # I wonder if this is handling transparency now, lol
        draw = ImageDraw.Draw(image)
        if stroke_width > 0:
            # Note that PIL doesn't seem to support alpha in stroke...interesting!
            stroke_width = stroke_width // 2
            stroke_fill = fill
            fill = (0, 0, 0, 0)
            draw.text(
                xy=(xy[0] - stroke_width, xy[1] - stroke_width),
                text=text_str,
                font=font,
                fill=fill,
                stroke_width=stroke_width,
                stroke_fill=stroke_fill)
        else:
            stroke_fill = (0, 0, 0, 0)
            draw.text(
                xy=xy,
                text=text_str,
                font=font,
                fill=fill,
                stroke_width=0,
                stroke_fill=stroke_fill)
    else:
        text_scala.draw_on_image(
            im_dest=image,
            pos=xy,
            text=text_str,
            font=_font_to_tuple(font),
            fill=fill,
            border_size=BORDER_DEFAULT,
            stroke_width=stroke_width)


def getsize(font: FreeTypeFont, text_str: str) -> Tuple[int, int]:
    """get the size of text"""
    if USE_PIL:
        # original behavior
        # size = font.getsize(text_str)

        # new behavior
        width, _ = font.getsize(text_str)
        ascent, descent = font.getmetrics()
        height = ascent + descent
        size = (width, height)

    else:
        info = text_scala.get_info(text_str, _font_to_tuple(font), 0, BORDER_DEFAULT)
        # Important note! Info height and width may not include some antialiasing pixels!
        # import numpy as np
        # column_sums = np.sum(img, axis=0)
        # present = np.where(column_sums > 0)[0]
        # calculated_width = (present[-1] - present[0]) + 1
        # print("metrics width:", info["width"])
        # print("calculated width:", calculated_width)
        size = (info["width"], info["height"])
    return size


def getoffset(font: FreeTypeFont, text_str: str) -> Tuple[int, int]:
    """get the offset of text"""
    if USE_PIL:
        offset = font.getoffset(text_str)
    else:
        # FOR NOW...because I don't think this is truly necessary
        offset = (0, 0)
    return offset


def getmetrics(font: Any) -> Tuple[int, int]:
    """get ascent and descent"""
    if USE_PIL:
        metrics = font.getmetrics()
    else:
        info = text_scala.get_info("Metrics", _font_to_tuple(font), 0, BORDER_DEFAULT)
        metrics = (info["ascent"], info["descent"])
    return metrics


def getleading(font: Any) -> int:
    """get leading"""
    if USE_PIL:
        leading = 0
    else:
        info = text_scala.get_info("Metrics", _font_to_tuple(font), 0, BORDER_DEFAULT)
        leading = info["leading"]
    return leading


def _font_to_tuple(font: Any) -> Tuple:
    """get a tuple of font information (used by text_scala)"""

    if isinstance(font, FreeTypeFont):
        font_face, font_style = font.getname()
        font_size = font.size
        font_style = font_style.lower()
    else:
        font_face, _ = font[0].getname()
        font_size = font[0].size
        font_style = font[1]

    return font_face, font_style, font_size
