"""

Test blimp_text

"""

# Copyright (c) 2020 Ben Zimmer. All rights resrved.

import os

from PIL import Image
import numpy as np

from symbols import blimp, blimp_text, debugutil


DEBUG_VISUALIZE = os.environ.get("DEBUG_VISUALIZE") == "true"


def test_blimp_text():
    """test text rendering"""
    # some of this might belong more in a test for blimp_text,
    # but I think that module will go away eventually.

    # TODO: totally transparent color instead of None

    font = blimp.load_font("Cinzel-Regular.ttf", 100)
    missing = (0, 0, 0, 0)  # note that this is ignored for text_scala mode

    args_list = [
        ("AVIARY", font, missing, 2, (0, 255, 255)),
        ("AVIARY", font, missing, 2, (0, 255, 255, 128)),
        ("AVIARY", font, (0, 255, 255), 0, missing),
        ("AVIARY", font, (0, 255, 255, 128), 0, missing),
    ]

    for args in args_list:
        for use_pil in [True, False]:
            print("~~~~~~~~~~")
            blimp_text.USE_PIL = use_pil
            size = blimp_text.getsize(font, args[0])
            # im_res = Image.new("RGBA", size, (0, 0, 0, 0))
            im_res = blimp.text_standard(*args)
            # blimp_text.text(im_res, (0, 0), *args)
            assert im_res.size == size
            assert im_res.mode == "RGBA"
            if DEBUG_VISUALIZE:
                debugutil.show(im_res, "test_text_standard " + str(use_pil))
            # TODO: save this in better place
            im_res.save("text_" + str(use_pil) + ".png")


def test_text_border():
    """test some issues related to borders in text, specifically
    how antialiasing pixels may reach beyond the start and end positions,
    and how borders are necessary to capture these pixels."""

    # The goal here is easy comparison by flipping back and forth in PyCharm.
    # TODO: also add programatic verification that font extends into border

    font = blimp.load_font("Cinzel-Regular.ttf", 200)
    missing = (0, 0, 0, 0)  # note that this is ignored for text_scala mode

    text = "AVIARY"
    text_color = (0, 255, 255, 128)
    text_stroke_width = 0
    text_stroke_color = missing
    border_size = 32

    def add_guides(img: np.ndarray):
        """add guides"""
        img[border_size, :, :] = (128, 128, 128, 255)
        img[:, border_size, :] = (128, 128, 128, 255)
        img[border_size + size[1], :, :] = (128, 128, 128, 255)
        img[:, border_size + size[0], :] = (128, 128, 128, 255)

    # we only care about scala mode here
    blimp_text.USE_PIL = False

    size = blimp_text.getsize(font, text)  # calculate expected size

    # get the text image using text_standard

    im_res = blimp.text_standard(
        text, font, text_color, text_stroke_width, text_stroke_color)

    assert size == im_res.size

    # extend the border and add guides

    im_res_expanded = blimp.expand_border(np.array(im_res), border_size, border_size)
    add_guides(im_res_expanded)
    im_res_expanded = Image.fromarray(im_res_expanded)

    # get the same image with borders and add guides

    im_res_border = blimp.text_standard_border(
        text, (border_size, border_size), font, text_color, text_stroke_width, text_stroke_color)
    im_res_border = np.array(im_res_border)
    add_guides(im_res_border)
    im_res_border = Image.fromarray(im_res_border)

    assert im_res_border.size == im_res_expanded.size

    # ~~~~ save stuff

    im_res.save("text_border_standard.png")
    im_res_expanded.save("text_border_standard_expanded.png")
    im_res_border.save("text_border_standard_border.png")
