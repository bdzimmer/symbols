"""

Stuff for refactoring text in blimp.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from typing import Tuple

from PIL import ImageDraw
from PIL.ImageFont import FreeTypeFont

from symbols import text_scala

USE_PIL = False

# TODO: THIS SHOULD NOT BE 0, 0
BORDER_DEFAULT = (0, 0)


def text(image, xy, text_str, font: FreeTypeFont, fill, stroke_width, stroke_fill):
    """draw text on an image, mutating it"""
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
        # note that fill works differently here than PIL
        # I think that's ok
        text_scala.draw_on_image(
            im=image,
            pos=xy,
            text=text_str,
            font=_font_to_tuple(font),
            fill=fill,
            border_size=BORDER_DEFAULT,
            stroke_width=stroke_width)


def getsize(font: FreeTypeFont, text_str: str) -> Tuple[int, int]:
    """get the size of text"""
    if USE_PIL:
        return font.getsize(text_str)
    else:
        _, info = text_scala.draw(text_str, _font_to_tuple(font), 0, BORDER_DEFAULT)
        return info["width"], info["height"]


def getoffset(font: FreeTypeFont, text_str: str) -> Tuple[int, int]:
    """get the offset of text"""
    if USE_PIL:
        return font.getoffset(text_str)
    else:
        # FOR NOW...because I don't think this is actually necessary
        return 0, 0


def getmetrics(font: FreeTypeFont) -> Tuple[int, int]:
    """get ascent and descent"""
    if USE_PIL:
        return font.getmetrics()
    else:
        _, info = text_scala.draw("Metrics", _font_to_tuple(font), 0, BORDER_DEFAULT)
        return info["ascent"], info["descent"]


def _font_to_tuple(font: FreeTypeFont):
    """aldfjhaldkjfhlajdh"""
    font_face, font_style = font.getname()
    font_size = font.size
    return font_face, font_style, font_size
