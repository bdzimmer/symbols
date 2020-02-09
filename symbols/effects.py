"""

Image effects and related utilities.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import cv2
import numpy as np


def I(func):
    """convert a channel-wise function to an image-wise function"""
    def wrap(im):
        r = im[:, :, 0]
        g = im[:, :, 1]
        b = im[:, :, 2]
        r, g, b = func(r, g, b)
        return np.stack([r, g, b], axis=2)
    return wrap


class C:
    """composable function"""

    def __init__(self, func):
        self.func = func

    def __call__(self, x):
        return self.func(x)

    def __matmul__(self, other):
        return C(lambda x: self(other(x)))


def grid(im):
    """grid / CRT like effects"""
    im = np.copy(im)

    # scanlines
    im[::8, :, :] = 0
    im[1::8, :, :] = 0
    im[:, :8, :] = 0
    im[:, 1::8, :] = 0

    return im


def glow(im, size, factor):
    """glow effect"""
    blurred = np.clip(cv2.GaussianBlur(im, (size, size), 0) * factor, 0, 255)
    return np.maximum(im, blurred)