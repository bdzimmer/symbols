"""

Stuff for refactoring text in blimp.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from typing import Tuple

from PIL import ImageDraw
from PIL.ImageFont import FreeTypeFont

from symbols import text_scala

USE_PIL = False

BORDER_DEFAULT = (16, 16)


def text(image, xy, text_str, font: FreeTypeFont, fill, stroke_width, stroke_fill):
    """draw text on an image, mutating it"""

    # pylint: disable=too-many-arguments

    # Note: text_scala does not support both fill and stroke at the same time.
    # If stroke_width > 0, fill is the color of the stroke.
    # Note that the if doing stroke, the fill argument is not used.
    # If stroke_width = 0, the stroke_fill argument is not used.

    if USE_PIL:
        draw = ImageDraw.Draw(image)
        draw.text(
            xy=xy,
            text=text_str,
            font=font,
            fill=fill,
            stroke_width=stroke_width,
            stroke_fill=stroke_fill)
    else:
        if stroke_width > 0:
            fill = stroke_fill
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
        _, info = text_scala.draw(text_str, _font_to_tuple(font), 0, BORDER_DEFAULT)
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


def getmetrics(font: FreeTypeFont) -> Tuple[int, int]:
    """get ascent and descent"""
    if USE_PIL:
        metrics = font.getmetrics()
    else:
        _, info = text_scala.draw("Metrics", _font_to_tuple(font), 0, BORDER_DEFAULT)
        metrics = (info["ascent"], info["descent"])
    return metrics


def _font_to_tuple(font):
    """aldfjhaldkjfhlajdh"""

    if isinstance(font, FreeTypeFont):
        font_face, font_style = font.getname()
        font_size = font.size
        font_style = font_style.lower()
    else:
        font_face, _ = font[0].getname()
        font_size = font[0].size
        font_style = font[1]

    return font_face, font_style, font_size
