"""
Imtegration tests for compositing functionality.
"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import os
import sys

import cv2
import numpy as np
from PIL import Image

from symbols import blimp

# reused stuff
RESOURCES_DIRNAME = [
    "C:/Ben/Google Drive/art",
    "/home/ben/Google Drive/art"][1]

DEBUG = True

# There's currently some issues with this set to false
FORCE_CUSTOM_KERNING = True

CANVAS_WIDTH = 3000
CANVAS_HEIGHT = 500

SCRATCH_DIRNAME = os.path.join("test_scratch", "composite")


def test_text():
    """Test various text use cases."""

    if DEBUG:
        os.makedirs(SCRATCH_DIRNAME, exist_ok=True)

    bg_filename = os.path.join(
        RESOURCES_DIRNAME,
        "unsplash",
        "bonnie-kittle-aQnyyf-4uZQ-unsplash.jpg")

    im_bg = cv2.imread(bg_filename)
    if im_bg is None:
        print("background image not found")
        sys.exit()

    im_bg = blimp.add_alpha(im_bg)

    # ~~~~ example 0: text with stroke

    layer_text = {
        "type": "text",
        "text": "PROVIDENCE",
        "font": "Orbitron-Bold.ttf",
        "size": 350,
        # TODO: alpha in text color is broken
        "color": (0, 0, 0),  # (0, 0, 0, 255),
        "force_custom_kerning": FORCE_CUSTOM_KERNING,
        "stroke_width": 3,
        "stroke_fill": (0, 0, 255, 255)
    }

    # TODO: appears that fill / stroke are not working together

    im_text, im_comp = render_and_composite([layer_text], RESOURCES_DIRNAME, im_bg)

    assert len(im_comp.shape) == 3
    assert im_comp.dtype == np.ubyte
    assert len(im_text.shape) == 3
    assert im_text.dtype == np.ubyte
    assert im_comp.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 4)

    if DEBUG:
        cv2.imwrite(os.path.join(SCRATCH_DIRNAME, "text_0.png"), im_text)
        cv2.imwrite(os.path.join(SCRATCH_DIRNAME, "comp_0.png"), im_comp)

    # ~~~~ example 1: text with double-sided glow

    # Also note, due to the way that borders are handled, these layers should line
    # up automatically no matter which size stroke is defined on each.
    # Not sure how borders factor in.

    layers_text = [
        {
            "type": "text",
            "text": "PROVIDENCE",
            "font": "Orbitron-Bold.ttf",
            "size": 350,
            "color": (40, 40, 40),  # alpha 255
            "force_custom_kerning": FORCE_CUSTOM_KERNING,
            "border_x": 32,
            "border_y": 32
        },
        {
            "type": "text",
            "text": "PROVIDENCE",
            "font": "Orbitron-Bold.ttf",
            "size": 350,
            "color": (0, 0, 0),  # alpha 0
            "stroke_width": 4,
            "stroke_fill": (255, 0, 255),  # alpha 255
            "force_custom_kerning": FORCE_CUSTOM_KERNING,
            "border_x": 32,
            "border_y": 32,
            "effects": [
                {
                    "type": "glow",
                    "dilate": 4,
                    "blur": 63,
                    "color": (255, 0, 255)
                }
            ]
        }
    ]

    im_black = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 4), dtype=np.uint8)
    im_black[:, :, 3] = 255
    im_text, im_comp = render_and_composite(layers_text, RESOURCES_DIRNAME, im_black)

    # TODO: there's a shift here that's incorrect

    if DEBUG:
        cv2.imwrite(os.path.join(SCRATCH_DIRNAME, "text_1.png"), im_text)
        cv2.imwrite(os.path.join(SCRATCH_DIRNAME, "comp_1.png"), im_comp)

    # ~~~~ example 2: text with inner glow

    # I thought this would be straightforward with mask / mask_onto,
    # but I was wrong. A project for another time.

    # ~~~~ example 3: masked text with outer glow

    # Note that because the glow is greated from edge detection on the masked
    # image, it's uneven due to the texture. To get an even glow, need another
    # layer. See the other examples.

    layer_text = {
        "type": "text",
        "text": "PROVIDENCE",
        "font": "Orbitron-Bold.ttf",
        "size": 350,
        # when masking, the text color should not matter and not show
        "color": (0, 0, 255),  # 255 alpha;
        "force_custom_kerning": FORCE_CUSTOM_KERNING,
        "border_x": 32,
        "border_y": 32,
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

    im_text, im_comp = render_and_composite([layer_text], RESOURCES_DIRNAME, im_bg)

    if DEBUG:
        cv2.imwrite(os.path.join(SCRATCH_DIRNAME, "text_3.png"), im_text)
        cv2.imwrite(os.path.join(SCRATCH_DIRNAME, "comp_3.png"), im_comp)


def render_and_composite(layers, resources_dirname, im_bg):
    """render a layer, apply effects, and composite against a background image"""

    # border_px = 64

    blimp.DEBUG = False
    # im_layer = blimp.render_layer(layer, resources_dirname)
    # im_layer = blimp.expand_border(im_layer, border_px, border_px)
    # for effect_idx, effect in enumerate(layer.get("effects", [])):
    #     im_layer = blimp.apply_effect(im_layer, effect, resources_dirname)
    #
    # # slice out the same sized chunk from the bg image
    # # and add alpha
    # im_bg_chunk = im_bg[0:im_layer.shape[0], 0:im_layer.shape[1], :]
    # im_bg_chunk = blimp.add_alpha(im_bg_chunk)
    #

    canvas_layer = {"type": "empty", "width": CANVAS_WIDTH, "height": CANVAS_HEIGHT}

    im_bg_chunk = im_bg[0:CANVAS_HEIGHT, 0:CANVAS_WIDTH]

    im_layer = blimp.assemble_group(
        [canvas_layer] + layers,
        CANVAS_WIDTH, CANVAS_HEIGHT,
        resources_dirname, True, False, False, None, None,
        [])

    # composite
    im_comp = Image.fromarray(im_bg_chunk)
    im_comp.alpha_composite(Image.fromarray(im_layer))
    im_comp = np.array(im_comp)

    # TODO: add some vertical lines to ensure that things are centered properly
    # TODO: seeing something like a one-pixel offset issue with glow

    return im_layer, im_comp
