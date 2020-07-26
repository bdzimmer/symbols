"""

Utilities for debugging.

"""

# Copryight (c) 2020 Ben Zimmer. All rights reserved.

import os

import cv2
import numpy as np
from PIL import Image

BG_COLOR = (80, 80, 80)
# BG_COLOR = (0, 0, 0)

WINDOW_STYLE = cv2.WINDOW_AUTOSIZE
# WINDOW_STYLE = cv2.WINDOW_KEEPRATIO


def show(img, title):
    """easily display an image for debugging"""

    if isinstance(img, Image.Image):
        img = np.array(img)

    _, _, n_channels = img.shape

    # print("img max before flatten:  ", np.max(img[:, :, 0:3]))

    if n_channels > 3:
        # print("flattening alpha")
        # print("alpha max before flatten:", np.max(img[:, :, 3]))
        img = flatten_alpha(img)

    # print("img max after flatten:  ", np.max(img[:, :, 0:3]))

    cv2.namedWindow(title, WINDOW_STYLE)
    cv2.imshow(title, img[:, :, [2, 1, 0]])

    while True:
        k = cv2.waitKey(50)
        if k > -1:
            break
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break
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

    _, _, n_channels1 = im1.shape
    n_channels2 = im2.shape[2]

    if n_channels1 > 3:
        print("flattening alpha")
        im1 = flatten_alpha(im1)
    if n_channels2 > 3:
        print("flattening alpha")
        im2 = flatten_alpha(im2)

    cv2.namedWindow(title, flags=WINDOW_STYLE)

    im_disp = np.concatenate((im1, im2), axis=1)
    cv2.imshow(title, im_disp[:, :, [2, 1, 0]])

    while True:
        k = cv2.waitKey(50)
        if k > -1:
            break
        if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyWindow(title)


def flatten_alpha(img: np.ndarray) -> np.ndarray:
    """paste an image with alpha onta a solid background"""

    # convert and prepare PIL images
    img = Image.fromarray(img)
    im_flat = Image.new("RGB", img.size, BG_COLOR)

    # paste using PIL paste
    im_flat.paste(img.convert("RGB"), (0, 0), img.split()[3])

    # return numpy array version
    return np.array(im_flat)


def save_image(img: Image, scratch_dirname: str, filename: str) -> None:
    """save an image"""
    os.makedirs(scratch_dirname, exist_ok=True)
    output_filename = os.path.join(scratch_dirname, filename)
    print(output_filename)
    img.save(output_filename)
