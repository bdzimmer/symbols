"""
Tests for compositing functionality.
"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import os
import unittest

import cv2
import numpy as np
from PIL import Image

import blimp    # TODO: refactor and import library funcs

DEBUG = True


class TestsComposite(unittest.TestCase):
    """Tests for compositing."""

    def test_alpha(self):
        """tests for alpha blending"""

        # render some text, then stick it on a background image
        # with some different additional effects

        border_px = 64
        text_color = [0, 0, 0, 255]  # [255, 255, 255, 255],

        bg_filename = os.path.join(
            # "/media/ben/Storage/Ben/Google Drive/art",
            "C:/Ben/Google Drive/art",
            "unsplash",
            "ian-keefe-2X89pjPktyA-unsplash.jpg")
        im_bg = cv2.imread(bg_filename)

        layer_text = {
            "type": "text",
            "text": "PROVIDENCE",
            "font": "Orbitron-Bold.ttf",
            "size": 250,
            "color": text_color,
            "stroke_width": 3,
            "stroke_fill": (0, 0, 255, 255),
            "effects": [
                {
                    "type": "glow",
                    "dilate": 4,
                    "blur": 31
                }
            ]
        }

        # render layer, expand borders, apply effects

        blimp.DEBUG = False
        im_text = blimp.render_layer(layer_text, None)
        im_text = blimp.expand_border(im_text, border_px, border_px)
        for effect_idx, effect in enumerate(layer_text.get("effects", [])):
            im_text = blimp.apply_effect(im_text, effect, None)

        # print(im_text.shape)

        # slice out the same sized chunk from the bg image
        # and add alpha
        im_bg_chunk = im_bg[0:im_text.shape[0], 0:im_text.shape[1], :]
        im_bg_chunk = blimp.add_alpha(im_bg_chunk)

        # composite
        comp = Image.fromarray(im_bg_chunk)
        comp.alpha_composite(Image.fromarray(im_text))
        comp = np.array(comp)

        # TODO: add some vertical lines to ensure that things are centered properly
        # TODO: seeing something like a one-pixel offset issue with glow

        if DEBUG:
            cv2.imwrite("text_0.png", im_text)
            cv2.imwrite("comp_0.png", comp)

        self.assertEqual(len(im_text.shape), 3)
        self.assertEqual(im_text.dtype, np.ubyte)

        self.assertEqual(len(comp.shape), 3)
        self.assertEqual(comp.dtype, np.ubyte)
        self.assertEqual(comp.shape, im_text.shape)
