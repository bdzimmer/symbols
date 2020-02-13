"""
Imtegration tests for compositing functionality.
"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import os
import unittest

import cv2
import numpy as np
from PIL import Image

import blimp    # TODO: refactor and import library funcs instead

# reused stuff
RESOURCES_DIRNAME = [
    "C:/Ben/Google Drive/art",
    "/media/ben/Storage/Ben/Google Drive/art"][0]

DEBUG = True


class TestsComposite(unittest.TestCase):
    """Tests for compositing."""

    def test_text(self):
        """Test various text use cases."""

        bg_filename = os.path.join(
            RESOURCES_DIRNAME,
            "unsplash",
            "bonnie-kittle-aQnyyf-4uZQ-unsplash.jpg")

        im_bg = cv2.imread(bg_filename)

        # ~~~~ example 0: text with stroke

        layer_text = {
            "type": "text",
            "text": "PROVIDENCE",
            "font": "Orbitron-Bold.ttf",
            "size": 350,
            "color": (0, 0, 0, 255),  # text_color,
            "stroke_width": 3,
            "stroke_fill": (0, 0, 255, 255)
        }

        im_text, im_comp = render_and_composite(layer_text, RESOURCES_DIRNAME, im_bg)

        self.assertEqual(len(im_comp.shape), 3)
        self.assertEqual(im_comp.dtype, np.ubyte)
        self.assertEqual(len(im_text.shape), 3)
        self.assertEqual(im_text.dtype, np.ubyte)
        self.assertEqual(im_comp.shape, im_text.shape)

        if DEBUG:
            cv2.imwrite("text_0.png", im_text)
            cv2.imwrite("comp_0.png", im_comp)

        # ~~~~ example 1: masked text with outer glow

        layer_text = {
            "type": "text",
            "text": "PROVIDENCE",
            "font": "Orbitron-Bold.ttf",
            "size": 350,
            "color": (0, 0, 255, 255),  # when masking, the text color should not matter and not show
            "effects": [
                {
                    "type": "mask_onto",
                    "layer": {
                        "type": "image",
                        "filename": "unsplash/austin-templeton-TWMnY0rtvoo-unsplash.jpg"
                    }
                },
                {
                    "type": "glow",
                    "dilate": 4,
                    "blur": 31,
                    "only": False
                }
            ]
        }

        im_text, im_comp = render_and_composite(layer_text, RESOURCES_DIRNAME, im_bg)

        if DEBUG:
            cv2.imwrite("text_1.png", im_text)
            cv2.imwrite("comp_1.png", im_comp)


def render_and_composite(layer, resources_dirname, im_bg):
    """render a layer, apply effects, and composite against a background image"""

    border_px = 64

    blimp.DEBUG = False
    im_layer = blimp.render_layer(layer, resources_dirname)
    im_layer = blimp.expand_border(im_layer, border_px, border_px)
    for effect_idx, effect in enumerate(layer.get("effects", [])):
        im_layer = blimp.apply_effect(im_layer, effect, resources_dirname)

    # slice out the same sized chunk from the bg image
    # and add alpha
    im_bg_chunk = im_bg[0:im_layer.shape[0], 0:im_layer.shape[1], :]
    im_bg_chunk = blimp.add_alpha(im_bg_chunk)

    # composite
    im_comp = Image.fromarray(im_bg_chunk)
    im_comp.alpha_composite(Image.fromarray(im_layer))
    im_comp = np.array(im_comp)

    # TODO: add some vertical lines to ensure that things are centered properly
    # TODO: seeing something like a one-pixel offset issue with glow

    return im_layer, im_comp
