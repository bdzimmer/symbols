"""

Utilities for debugging.

"""

# Copryight (c) 2020 Ben Zimmer. All rights reserved.

import sys

import cv2
import numpy as np
from PIL import Image

BG_COLOR = (80, 80, 80)


def show(img, title):
    """easily display an image for debugging"""

    if isinstance(img, Image.Image):
        img = np.array(img)

    im_height, im_width, n_channels = img.shape
    if n_channels > 3:
        print("flattening alpha")
        img = flatten_alpha(img)

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)

    cv2.imshow(title, img[:, :, [2, 1, 0]])
    cv2.waitKey(-1)
    cv2.destroyWindow(title)


def show_comparison(im1, im2, title):
    """easily display before and after images for debugging purposes in tests"""

    # Normally I don't like this kind of polymorphism, but I'll make
    # an exception for tests.

    if isinstance(im1, Image.Image):
        im1 = np.array(im1)

    if isinstance(im2, Image.Image):
        im2 = np.array(im2)

    im1 = np.array(im1, dtype=np.ubyte)
    im2 = np.array(im2, dtype=np.ubyte)

    im_height, im_width, n_channels1 = im1.shape
    n_channels2 = im2.shape[2]

    if n_channels1 > 3:
        print("flattening alpha")
        im1 = flatten_alpha(im1)
    if n_channels2 > 3:
        print("flattening alpha")
        im2 = flatten_alpha(im2)

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    # cv2.resizeWindow(title, im_width * 2, im_height)

    im_disp = np.concatenate((im1, im2), axis=1)

    cv2.imshow(title, im_disp[:, :, [2, 1, 0]])
    cv2.waitKey(-1)
    cv2.destroyWindow(title)


def flatten_alpha(img: np.array) -> np.array:
    """paste an image with alpha onta a solid background"""

    # convert and prepare PIL images
    img = Image.fromarray(img)
    im_flat = Image.new("RGB", img.size, BG_COLOR)

    # paste using PIL paste
    im_flat.paste(img.convert("RGB"), (0, 0), img.split()[3])

    # return numpy array version
    return np.array(im_flat)
