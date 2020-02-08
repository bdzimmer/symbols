"""
Driver / test program from image effects.
"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import os
from functools import partial as P
import random
import time

import cv2
import numpy as np

from symbols.effects import I, C


def main():
    """main program"""

    start_time = time.time()

    input_dirname = "C:/Ben/Google Drive/art"
    input_filename = os.path.join(
        input_dirname,
        "unsplash",
        "nik-macmillan-gEkL3UfB3qw-unsplash.jpg")

    random.seed(1)

    # output_filename = "render.png"
    output_filename = "render.jpg"  # for faster output

    im_org = cv2.imread(input_filename)

    # scale down
    im_org = cv2.resize(im_org, None, None, 0.25, 0.25, cv2.INTER_LINEAR)

    # convert to RGB
    im = cv2.cvtColor(im_org, cv2.COLOR_BGR2RGB)

    uint8 = P(np.array, dtype=np.uint8)
    clip = P(np.clip, a_min=0, a_max=255)

    def pow(c, exp):
        c_scaled = c / 255.0
        return np.array(np.power(c_scaled, exp) * 255.0, dtype=np.uint8)

    def etherize(r, g, b):
        r = pow(r, 1.5) * 0.3
        g = pow(g, 2.5) * 5.0
        b = pow(b + r, 3.0) * 20.0
        return r, g, b

    def monochrome(r, g, b):
        x = (r + 1.5 * g + b) / 3.5
        x = pow(x, 1.5) * 2.0
        return x, x, x

    def mean(x):
        return lambda y: (x + y) / 2.0

    def shift(x, y):
        mat = np.array([[1.0, 0.0, x], [0.0, 1.0, y]])
        return P(cv2.warpAffine,  M=mat, dsize=None)

    mono_and_shift = C(uint8) @ clip @ shift(3, 0) @ I(monochrome)
    im_mono = mono_and_shift(im)

    ether_abber = C(uint8) @ clip @ mean(im_mono) @ I(etherize)
    im = ether_abber(im)

    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    im_both = np.concatenate((im_org, im), 1)
    cv2.imwrite(output_filename, im_both)

    end_time = time.time() - start_time
    print("total time:", round(end_time, 3), "sec")


if __name__ == "__main__":
    main()
