"""

Test blimp_text

"""

# Copyright (c) 2020 Ben Zimmer. All rights resrved.

from typing import Tuple
import os

from PIL import Image
import numpy as np

from symbols import blimp, blimp_text, debugutil


DEBUG = True
DEBUG_VISUALIZE = os.environ.get("DEBUG_VISUALIZE") == "true"
SCRATCH_DIRNAME = os.path.join("test_scratch", "text_border")


def test_blimp_text():
    """test text rendering"""
    # some of this might belong more in a test for blimp_text,
    # but I think that module will go away eventually.

    # TODO: totally transparent color instead of None

    font = blimp.load_font("Cinzel-Regular.ttf", 100)
    missing = (0, 0, 0, 0)  # note that this is ignored for text_scala mode

    args_list = [
        ("AVIARY", (0, 0), font, missing, 2, (0, 255, 255)),
        ("AVIARY", (0, 0), font, missing, 2, (0, 255, 255, 128)),
        ("AVIARY", (0, 0), font, (0, 255, 255), 0, missing),
        ("AVIARY", (0, 0), font, (0, 255, 255, 128), 0, missing),
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
    # font = blimp.load_font("consola.ttf", 200)
    missing = (0, 0, 0, 0)  # note that this is ignored for text_scala mode

    text = "AVIARY"
    # text_color = (0, 255, 255, 128)
    text_color = (0, 255, 255, 255)
    text_stroke_width = 0
    text_stroke_color = missing
    border_size = 32
    text_kern_add = 32

    def add_guides(img: np.ndarray, size: Tuple):
        """add guides"""
        img[border_size, :, :] = (128, 128, 128, 255)
        img[:, border_size, :] = (128, 128, 128, 255)
        img[border_size + size[1], :, :] = (128, 128, 128, 255)
        img[:, border_size + size[0], :] = (128, 128, 128, 255)

    for use_pil in [False, True]:

        blimp_text.USE_PIL = use_pil

        # ~~~~ standard text ~~~~
        print()
        print(f"standard_text use_pil={use_pil}")
        print("----------")

        # get the text image using text_standard

        im_res = blimp.text_standard(
            text, (0, 0), font, text_color, text_stroke_width, text_stroke_color)

        # extend the border and add guides

        im_res_expanded = blimp.expand_border(np.array(im_res), border_size, border_size)
        add_guides(im_res_expanded, im_res.size)
        im_res_expanded = Image.fromarray(im_res_expanded)

        # get the same image with borders and add guides

        im_res_border = blimp.text_standard(
            text, (border_size, border_size), font, text_color, text_stroke_width, text_stroke_color)
        im_res_border = np.array(im_res_border)
        add_guides(im_res_border, im_res.size)
        im_res_border = Image.fromarray(im_res_border)

        # ~~~~ custom kerning text ~~~~
        print()
        print(f"custom_kerning use_pil={use_pil}")
        print("----------")

        # get the text image using text_standard

        im_custom = blimp.text_custom_kerning(
            text, (0, 0), font, text_color, text_stroke_width, text_stroke_color, text_kern_add, False)

        # extend the border and add guides

        im_custom_expanded = blimp.expand_border(np.array(im_custom), border_size, border_size)
        add_guides(im_custom_expanded, im_custom.size)
        im_custom_expanded = Image.fromarray(im_custom_expanded)

        # get the same image with borders and add guides

        im_custom_border = blimp.text_custom_kerning(
            text, (border_size, border_size), font, text_color, text_stroke_width, text_stroke_color, text_kern_add, False)
        im_custom_border = np.array(im_custom_border)
        add_guides(im_custom_border, im_custom.size)
        im_custom_border = Image.fromarray(im_custom_border)

        # assertions
        # (we only care about these for certain methods)

        if not use_pil:
            assert im_res_border.size == im_res_expanded.size
            assert im_custom_border.size == im_custom_expanded.size
            # TODO: assert some things about running over the borders

        # ~~~~ save stuff ~~~~

        method_str = "scala" if use_pil == False else "pil"

        _debug_save_image(im_res, f"{method_str}_standard.png")
        _debug_save_image(im_res_expanded, f"{method_str}_standard_expanded.png")
        _debug_save_image(im_res_border, f"{method_str}_standard_border.png")

        _debug_save_image(im_custom, f"{method_str}_custom.png")
        _debug_save_image(im_custom_expanded, f"{method_str}_custom_expanded.png")
        _debug_save_image(im_custom_border, f"{method_str}_custom_border.png")


def _debug_save_image(img: Image, filename: str):
    """save an image"""
    if DEBUG:
        os.makedirs(SCRATCH_DIRNAME, exist_ok=True)
        output_filename = os.path.join(SCRATCH_DIRNAME, filename)
        img.save(output_filename)
