"""

Image conversion functions.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from typing import Tuple

import numpy as np
from PIL import Image


# Some functions for colorizing single channel black and white image (PIL "L" mode)
# or the alpha channels of text_scala output.

# ~~~~ function from text_scala


def colorize(img: np.ndarray, color: Tuple) -> np.ndarray:
    """colorize a single-channel (alpha) image into a 4-channel RGBA image"""

    # ensure color to RGBA
    if len(color) == 3:
        color = (color[0], color[1], color[2], 255)

    # created result image filled with solid "color"
    res = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.ubyte)
    res[:, :, 0:4] = color

    # scale the alpha component by the image
    # (this comes into play if "color" has alpha < 255)
    res[:, :, 3] = color[3] / 255.0 * img

    # set the RGB of completely transparent pixels to zero
    res[res[:, :, 3] == 0, 0:3] = (0, 0, 0)

    return res


# ~~~~ function the old text module
# pretty much the only difference between these is order of operations
# in scaling of alpha. Could programatically verify that both do the
# same thing.

def l_to_rgba(img: np.ndarray, color: Tuple) -> np.ndarray:
    """create a colorized transparent image from black and white"""

    # create result image filled with solid "color"
    height, width = img.shape
    solid = Image.new("RGBA", (width, height), color)
    res = np.array(solid)

    # scale the alpha component by the image
    # (this comes into play if "color" has alpha < 255)
    res[:, :, 3] = res[:, :, 3] * (img / 255.0)

    # set the RGB of completely transparent pixels to zero
    res[res[:, :, 3] == 0, 0:3] = (0, 0, 0)

    return res
