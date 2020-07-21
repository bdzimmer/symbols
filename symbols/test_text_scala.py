"""

Test text functions.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import os

from PIL import Image

from symbols import text_scala, debugutil

DEBUG = True
DEBUG_VISUALIZE = os.environ.get("DEBUG_VISUALIZE") == "true"
SCRATCH_DIRNAME = os.path.join("test_scratch", "text_scala")


def test_draw():
    """test draw"""
    font = ("Cinzel", "plain", 64)
    img, info = text_scala.draw("AVIARY", font, 0, (32, 32))

    assert info["ascent"] == 63
    assert info["descent"] == 24
    assert info["width"] == 235
    assert info["height"] == 87
    assert info["borderX"] == 32
    assert info["borderY"] == 32
    assert "stroke" not in info

    assert img.shape == (151, 299, 4)

    _debug_save_image(Image.fromarray(img), "text_scala_0.png")

    if DEBUG_VISUALIZE:
        debugutil.show(img, "draw")


def test_draw_on_image():
    """test draw_on_image"""
    font = ("Cinzel", "plain", 64)
    img = Image.new("RGBA", (640, 480), (0, 0, 0, 0))
    img_org = img.copy()
    info = text_scala.draw_on_image(
        img, (64, 64), "AVIARY", font, (0, 0, 255), 1, (32, 32))

    assert info["ascent"] == 63
    assert info["descent"] == 24
    assert info["width"] == 235
    assert info["height"] == 87
    assert info["borderX"] == 32
    assert info["borderY"] == 32
    assert info["stroke"] == 1.0

    _debug_save_image(img, "text_scala_1.png")

    if DEBUG_VISUALIZE:
        debugutil.show_comparison(img_org, img, "draw_on_image")


def test_draw_multiline():
    """test draw_multiline"""

    paragraphs = [
        "  {b}Lorem {b}ipsum {i}dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        " ",
        "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.",
        "Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.",
        "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    ]

    font = ("Play", "plain", 64)
    img = text_scala.draw_multiline(
        paragraphs,
        font, (32, 32), (1920, 1080),
        True)

    assert img.shape == (1080 + 32 * 2, 1920 + 32 * 2, 4)

    _debug_save_image(Image.fromarray(img), "text_scala_2.png")


def _debug_save_image(img: Image, filename: str):
    """save an image"""
    if DEBUG:
        os.makedirs(SCRATCH_DIRNAME, exist_ok=True)
        output_filename = os.path.join(SCRATCH_DIRNAME, filename)
        img.save(output_filename)
