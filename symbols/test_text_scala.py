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
    assert info["stroke"] == 0.0

    assert img.shape == (151, 299, 4)

    _debug_save_image(Image.fromarray(img), "text_scala_0.png")

    if DEBUG_VISUALIZE:
        im_show = Image.new("RGB", (img.shape[1], img.shape[0]), (0, 0, 0))
        img = Image.fromarray(img)
        im_show.paste(img.convert("RGB"), (0, 0), img.split()[3])
        debugutil.show(im_show, "draw")


def test_draw_on_image():
    """test draw_on_image"""
    font = ("Cinzel", "plain", 64)
    img = Image.new("RGBA", (640, 480), (0, 0, 0))
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


def _debug_save_image(img: Image, filename: str):
    """save an image"""
    if DEBUG:
        os.makedirs(SCRATCH_DIRNAME, exist_ok=True)
        output_filename = os.path.join(SCRATCH_DIRNAME, filename)
        img.save(output_filename)
