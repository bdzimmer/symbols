"""

Utilities for debugging.

"""

# Copryight (c) 2020 Ben Zimmer. All rights reserved.

import cv2
import numpy as np
from PIL.Image import Image


def show(img, title):
    """easily display an image for debugging"""

    if isinstance(img, Image):
        img = np.array(img)

    im_height, im_width, _ = img.shape

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, im_width, im_height)

    cv2.imshow(title, img)
    cv2.waitKey(-1)
    cv2.destroyWindow(title)


def show_comparison(im1, im2, title):
    """easily display before and after images for debugging purposes in tests"""

    # Normally I don't like this kind of polymorphism, but I'll make
    # an exception for tests.

    if isinstance(im1, Image):
        im1 = np.array(im1)

    if isinstance(im2, Image):
        im2 = np.array(im2)

    im_height, im_width, _ = im1.shape

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, im_width * 2, im_height)

    cv2.imshow(title, np.concatenate((im1, im2), axis=1))
    cv2.waitKey(-1)
    cv2.destroyWindow(title)
