"""

Tests for image effects.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import os

from PIL import Image
import numpy as np

from symbols import effects, text_scala, debugutil, blimp

DEBUG_VISUALIZE = os.environ.get("DEBUG_VISUALIZE") == "true"


def test_I():  # pylint: disable=invalid-name
    """test I"""

    def sum_channels(r_ch, g_ch, b_ch):
        """helper"""
        total = r_ch * 1.0 + g_ch * 1.0 + b_ch * 1.0
        return total / 3, total / 3, total / 3

    im_test = _get_image()[:, :, 0:3]
    im_res = effects.I(sum_channels)(im_test)

    assert im_test.shape == im_res.shape
    assert np.allclose(im_res[:, :, 0], im_res[:, :, 1])
    assert np.allclose(im_res[:, :, 1], im_res[:, :, 2])

    if DEBUG_VISUALIZE:
        debugutil.show_comparison(im_test, im_res, "test_I")


def test_C():  # pylint: disable=invalid-name
    """test C"""
    add_one = effects.C(lambda x: x + 1)

    def add_two(num):
        """add two"""
        return num + 2

    add_six = add_one @ add_two @ (lambda x: x + 3)

    assert add_six(1) == 7


def test_grid():
    """test grid"""

    im_test = _get_image()
    im_res = effects.grid(im_test)

    # test a couple of rows, not all of them
    assert np.allclose(im_res[0, :, :], 0.0)
    assert np.allclose(im_res[:, 0, :], 0.0)
    assert np.allclose(im_res[1, :, :], 0.0)
    assert np.allclose(im_res[:, 1, :], 0.0)

    if DEBUG_VISUALIZE:
        debugutil.show_comparison(im_test, im_res, "test_grid")


def test_glow():
    """test glow"""
    im_test = _get_image()
    im_res = effects.glow(im_test, 31, 1.0)
    assert im_test.shape == im_res.shape

    if DEBUG_VISUALIZE:
        debugutil.show_comparison(im_test, im_res, "test_glow")

    # import cv2
    # cv2.imwrite("glow.png", im_res)


def test_glow_alpha():
    """test glow_alpha"""
    im_test = _get_image()
    im_test = blimp.add_alpha(im_test)
    im_res = effects.glow_alpha(im_test, 31, 1.5)
    assert im_test.shape == im_res.shape

    if DEBUG_VISUALIZE:
        im_black = Image.new("RGB", (384, 384), (0, 0, 0, 255))
        im1 = im_black.copy()
        im2 = im_black.copy()
        im_test = Image.fromarray(im_test)
        im_res = Image.fromarray(im_res)
        im1.paste(im_test.convert("RGB"), (0, 0), im_test.split()[3])
        im2.paste(im_res.convert("RGB"), (0, 0), im_res.split()[3])
        debugutil.show_comparison(im1, im2, "test_glow_alpha")


def _get_image():
    """get a test image for various effects"""
    font = ("Cinzel", "plain", 256)
    stroke_width = 0
    img = Image.new("RGB", (384, 384), (0, 0, 0))
    text_scala.draw_on_image(
        img, (96, 0), "Z", font, (0, 255, 255), stroke_width, (32, 32))
    return np.array(img)
