"""

Image effects and related utilities.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import cv2
import numpy as np
from PIL import Image


def I(func):  # pylint: disable=invalid-name
    """convert a channel-wise function to an image-wise function"""
    def wrap(img):
        r_ch = img[:, :, 0]
        g_ch = img[:, :, 1]
        b_ch = img[:, :, 2]
        r_ch, g_ch, b_ch = func(r_ch, g_ch, b_ch)
        return np.stack([r_ch, g_ch, b_ch], axis=2)
    return wrap


class C:  # pylint: disable=invalid-name
    """composable function"""

    def __init__(self, func):
        self.func = func

    def __call__(self, arg):
        return self.func(arg)

    def __matmul__(self, other):
        return C(lambda x: self(other(x)))


def grid(img):
    """simple grid / CRT like effect"""
    img = np.copy(img)

    # scanlines
    img[::8, :, :] = 0
    img[1::8, :, :] = 0
    img[:, :8, :] = 0
    img[:, 1::8, :] = 0

    return img


def grid_var(img, width, offset):
    """simple grid / CRT like effect"""
    img = np.copy(img)

    # scanlines
    img[offset::width, :, :] = 0
    img[:, offset::width, :] = 0

    return img


def glow(img, size, factor):
    """glow effect"""
    blurred = np.clip(cv2.GaussianBlur(img, (size, size), 0) * factor, 0, 255)
    return np.maximum(img, blurred)


def glow_alpha(img, size, factor):
    """glow effect that handles alpha transparency"""
    blurred = np.clip(cv2.GaussianBlur(img, (size, size), 0) * factor, 0, 255)

    # alpha-composite instead of simple maximum
    # return np.maximum(im, blurred)
    return np.array(Image.alpha_composite(
        Image.fromarray(np.array(img, dtype=np.uint8)),
        Image.fromarray(np.array(blurred, dtype=np.uint8))
    ))


def chromatic_abberation(img: np.array, shift: int) -> np.array:
    """chromatic abberation"""

    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    green = np.roll(green, -shift, axis=1)
    blue = np.roll(blue, shift, axis=1)

    return np.stack([red, green, blue], axis=2)


def slats_vertical(img: np.array, shifts: np.array) -> np.array:
    """shift vertical strips of the image"""
    n_slats = shifts.shape[0]
    slat_width = int(img.shape[1] / n_slats)
    res = np.zeros(img.shape, dtype=img.dtype)
    for idx in range(n_slats):
        start_x = idx * slat_width
        slat = img[:, start_x:(start_x + slat_width)]
        slat = np.roll(slat, shifts[idx], axis=0)
        res[:, start_x:(start_x + slat_width)] = slat
    return res


