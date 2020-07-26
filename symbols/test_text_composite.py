"""
Imtegration tests for compositing functionality.
"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import os
import sys

import cv2
import numpy as np

from symbols import blimp, blimp_util, blimp_text

# reused stuff
RESOURCES_DIRNAME = [
    "C:/Ben/Google Drive/art",
    "/home/ben/Google Drive/art"][1]

DEBUG = True

FORCE_CUSTOM_KERNING = True
USE_PIL = False
TRIM_X = False

CANVAS_WIDTH = 3000
CANVAS_HEIGHT = 500

SCRATCH_DIRNAME = os.path.join("test_scratch", "text_composite")


def test_text():
    """Test various text use cases."""

    blimp_text.USE_PIL = USE_PIL

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
    im_bg = im_bg[0:CANVAS_HEIGHT, 0:CANVAS_WIDTH, :]

    # ~~~~

    print()
    print("~~~~ example 0: plain old text, no stroke")

    layer_text = {
        "type": "text",
        "text": "PROVIDENCE",
        "font": "Orbitron-Bold.ttf",
        "size": 350,
        "color": (0, 0, 0),
        "force_custom_kerning": FORCE_CUSTOM_KERNING,
        "x": 64,
        "y": 64,
        "trim_x": TRIM_X
    }

    im_text, im_comp = blimp_util.render_and_composite([layer_text], RESOURCES_DIRNAME, im_bg)

    assert len(im_comp.shape) == 3
    assert im_comp.dtype == np.ubyte
    assert len(im_text.shape) == 3
    assert im_text.dtype == np.ubyte
    assert im_comp.shape == (CANVAS_HEIGHT, CANVAS_WIDTH, 4)

    if DEBUG:
        cv2.imwrite(os.path.join(SCRATCH_DIRNAME, "text_0.png"), im_text)
        cv2.imwrite(os.path.join(SCRATCH_DIRNAME, "comp_0.png"), im_comp)

    # ~~~~

    print()
    print("~~~~ example 1: two layers of stroke")
    # (for verifying that stroke lines up with text)

    layers_text = [
        {
            "type": "text",
            "text": "PROVIDENCE",
            "font": "Orbitron-Bold.ttf",
            "size": 350,
            "color": (0, 0, 0),  # experiment with and without transparency!
            "force_custom_kerning": FORCE_CUSTOM_KERNING,
            # "stroke_width": 3,
            "x": 64,
            "y": 64,
            "trim_x": TRIM_X
        },
        {
            "type": "text",
            "text": "PROVIDENCE",
            "font": "Orbitron-Bold.ttf",
            "size": 350,
            "color": (255, 255, 255, 128),  # experiment with and without transparency!
            "force_custom_kerning": FORCE_CUSTOM_KERNING,
            "stroke_width": 16,
            "x": 64,
            "y": 64,
            "trim_x": TRIM_X,
            "border_x": 64
        }
    ]

    im_text, im_comp = blimp_util.render_and_composite(layers_text, RESOURCES_DIRNAME, im_bg)

    if DEBUG:
        cv2.imwrite(os.path.join(SCRATCH_DIRNAME, "text_1.png"), im_text)
        cv2.imwrite(os.path.join(SCRATCH_DIRNAME, "comp_1.png"), im_comp)

    # ~~~~

    print()
    print("~~~~ example 2: text with double-sided glow")

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
            "x": 64,
            "y": 64,
            "trim_x": TRIM_X,
            "border_x": 32,
            "border_y": 32
        },
        {
            "type": "text",
            "text": "PROVIDENCE",
            "font": "Orbitron-Bold.ttf",
            "size": 350,
            "color": (255, 0, 255),  # alpha 255
            "stroke_width": 4,
            "force_custom_kerning": FORCE_CUSTOM_KERNING,
            "x": 64,
            "y": 64,
            "trim_x": TRIM_X,
            "border_x": 32,
            "border_y": 32,
            "effects": [
                {
                    "type": "glow",
                    "dilate": 4,
                    "blur": 63,
                    "color": (255, 0, 255)
                }
            ],
            # This should work...why doesn't it?
            # "mask":  {
            #     "type": "text",
            #     "text": "PROVIDENCE",
            #     "font": "Orbitron-Bold.ttf",
            #     "size": 350,
            #     "color": (255, 255, 255),  # alpha 255
            #     "force_custom_kerning": FORCE_CUSTOM_KERNING,
            #     "x": 64,
            #     "y": 64,
            #     "trim_x": False,
            #     "border_x": 32,
            #     "border_y": 32
            # }
        }
    ]

    im_black = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH, 4), dtype=np.uint8)
    im_black[:, :, 3] = 255
    im_text, im_comp = blimp_util.render_and_composite(layers_text, RESOURCES_DIRNAME, im_black)

    if DEBUG:
        cv2.imwrite(os.path.join(SCRATCH_DIRNAME, "text_2.png"), im_text)
        cv2.imwrite(os.path.join(SCRATCH_DIRNAME, "comp_2.png"), im_comp)

    # ~~~~

    print()
    print("~~~~ example 3: masked text with outer shadow")

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
        "x": 64,
        "y": 64,
        "trim_x": TRIM_X,
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

    im_text, im_comp = blimp_util.render_and_composite([layer_text], RESOURCES_DIRNAME, im_bg)

    if DEBUG:
        cv2.imwrite(os.path.join(SCRATCH_DIRNAME, "text_3.png"), im_text)
        cv2.imwrite(os.path.join(SCRATCH_DIRNAME, "comp_3.png"), im_comp)

    # ~~~~ example 4: text with inner glow

    # I thought this would be straightforward with mask / mask_onto,
    # but I was wrong. A project for another time.
