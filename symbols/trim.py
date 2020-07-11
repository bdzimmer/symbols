"""

Functions for borders and trimming.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from typing import Tuple

import numpy as np
from PIL import Image


def expand_border(image: np.ndarray, border_x: int, border_y: int) -> np.ndarray:
    """add a border to an image"""

    # TODO: get rid of this conversion from image to ndarray...not necessary at all
    res = Image.new(
        "RGBA",
        (image.shape[1] + 2 * border_x, image.shape[0] + 2 * border_y),
        (0, 0, 0, 0),  # (255, 255, 255, 0)
    )

    res = np.array(res)
    lim_y = res.shape[0] - border_y if border_y > 0 else res.shape[0]
    lim_x = res.shape[1] - border_x if border_x > 0 else res.shape[1]

    res[border_y:lim_y, border_x:lim_x] = image
    return res


def expand_down_right(image: np.ndarray, new_x: int, new_y: int) -> np.ndarray:
    """expand an image in the positive x and y direction"""

    res = Image.new(
        "RGBA",
        (new_x, new_y),
        # (255, 255, 255, 0),
        (0, 0, 0, 0)
    )
    res = np.array(res)
    res[0:image.shape[0], 0:image.shape[1]] = image
    return res


def find_trim_x_indices(img: np.ndarray) -> Tuple[int, int]:
    """Given an image, find the start and end indices to slice to
    only keep columns with visible pixels"""

    filled_idxs = np.where(np.sum(img[:, :, 3] > 0, axis=0))[0]
    start_x = filled_idxs[0]
    end_x = filled_idxs[-1] + 1

    return start_x, end_x


def trim(
        layer_image: np.ndarray,
        layer_x: int,
        layer_y: int,
        canvas_width: int,
        canvas_height: int) -> Tuple[np.ndarray, int, int]:

    """trim the layer to fit the canvas"""

    start_x = 0
    end_x = layer_image.shape[1]
    start_y = 0
    end_y = layer_image.shape[0]

    if layer_x < 0:
        start_x = 0 - layer_x
        layer_x = 0
    if layer_x + end_x > canvas_width:
        end_x = start_x + canvas_width

    if layer_y < 0:
        start_y = 0 - layer_y
        layer_y = 0
    if layer_y + end_y > canvas_height:
        end_y = start_y + canvas_height

    return layer_image[start_y:end_y, start_x:end_x, :], layer_x, layer_y
