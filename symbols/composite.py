"""

Various compositing functions.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import numpy as np


def alpha_blend(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Alpha blending."""

    src_rgb = src[:, :, 0:3]
    dst_rgb = dst[:, :, 0:3]
    src_a = src[:, :, 3:4] / 255.0
    dst_a = dst[:, :, 3:4] / 255.0

    out_a = src_a + dst_a * (1.0 - src_a)
    out_rgb = src_rgb * src_a + dst_rgb * dst_a * (1.0 - src_a)
    out_rgb[out_a[:, :, 0] > 0.0] = (out_rgb / out_a)[out_a[:, :, 0] > 0.0]
    out_rgb[out_a[:, :, 0] == 0.0] = 0.0

    res = np.concatenate(
        (np.array(out_rgb, dtype=np.uint8),
         np.array(out_a * 255, dtype=np.uint8)),
        axis=2)

    return res


def additive_blend(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Additive blending."""

    src_rgb = src[:, :, 0:3]
    dst_rgb = dst[:, :, 0:3]
    src_a = src[:, :, 3:4] / 255.0
    dst_a = dst[:, :, 3:4] / 255.0

    out_a = np.clip(src_a + dst_a, 0.0, 1.0)
    out_rgb = np.clip((src_rgb * src_a + dst_rgb * dst_a) / out_a, 0.0, 255.0)

    res = np.concatenate(
        (np.array(out_rgb, dtype=np.uint8),
         np.array(out_a * 255, dtype=np.uint8)),
        axis=2)

    return res
