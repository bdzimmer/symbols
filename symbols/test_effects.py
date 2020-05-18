"""

Tests for image effects.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import numpy as np

from symbols import effects


def test_I():  # pylint: disable=invalid-name
    """test I"""

    def sum_channels(r_ch, g_ch, b_ch):
        """helper"""
        total = r_ch + g_ch + b_ch
        return total / 3, total / 3, total / 3

    im_size = (64, 128, 3)
    im_test = np.ones(im_size)
    im_res = effects.I(sum_channels)(im_test)

    assert im_test.shape == im_res.shape
    assert np.allclose(im_res[:, :, 0], im_res[:, :, 1])
    assert np.allclose(im_res[:, :, 1], im_res[:, :, 2])
    assert np.allclose(im_res, 1.0)


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

    im_size = (64, 128, 3)
    im_test = np.ones(im_size)
    im_res = effects.grid(im_test)

    # test a couple of rows, not all of them
    assert np.allclose(im_res[0, :, :], 0.0)
    assert np.allclose(im_res[:, 0, :], 0.0)
    assert np.allclose(im_res[1, :, :], 0.0)
    assert np.allclose(im_res[:, 1, :], 0.0)


def test_glow():
    """test glow"""
    im_size = (64, 128, 3)
    im_test = np.ones(im_size)
    im_res = effects.glow(im_test, 7, 1.0)
    assert im_test.shape == im_res.shape


def test_glow_alpha():
    """test glow_alpha"""
    im_size = (64, 128, 4)
    im_test = np.ones(im_size)
    im_res = effects.glow_alpha(im_test, 7, 1.0)
    assert im_test.shape == im_res.shape
