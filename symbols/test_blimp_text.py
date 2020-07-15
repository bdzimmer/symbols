"""

Test blimp_text

"""

# Copyright (c) 2020 Ben Zimmer. All rights resrved.

import os

from PIL import Image
import numpy as np

from symbols import blimp, blimp_text, debugutil, trim


DEBUG = True
DEBUG_VISUALIZE = os.environ.get("DEBUG_VISUALIZE") == "true"
SCRATCH_DIRNAME = os.path.join("test_scratch", "text_border")


def test_blimp_text():
    """test text rendering"""
    # some of this might belong more in a test for blimp_text,
    # but I think that module will go away eventually.

    font = blimp.load_font("Cinzel-Regular.ttf", 100)

    args_list = [
        ("AVIARY", (0, 0), font, (0, 255, 255), 2),
        ("AVIARY", (0, 0), font, (0, 255, 255, 128), 2),
        ("AVIARY", (0, 0), font, (0, 255, 255), 0),
        ("AVIARY", (0, 0), font, (0, 255, 255, 128), 0),
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
            # im_res.save("text_" + str(use_pil) + ".png")


def test_text_border():
    """test some issues related to borders in text, specifically
    how antialiasing pixels may reach beyond the start and end positions,
    and how borders are necessary to capture these pixels."""

    # pylint: disable=too-many-locals
    # pylint: disable=too-many-statements

    # The goal here is easy comparison by flipping back and forth in PyCharm.
    # There is also some programatic verification that font serifs extend
    # into borders for certain configurations.

    # a few fonts for testing different issues:

    # serifs overflow size
    font = blimp.load_font("Cinzel-Regular.ttf", 200)

    # fixed-width
    # font = blimp.load_font("consola.ttf", 200)

    # gaps between start position and font pixels
    # font = blimp.load_font("Orbitron-Bold.ttf", 200)

    text = "AVIARY"
    # text = "axoxo"
    text_color = (0, 255, 255, 128)
    # text_color = (0, 255, 255, 255)
    text_stroke_width = 0
    border_size = 32
    text_kern_add = 0
    debug_lines = False  # note that enabling this will disable color tests

    for use_pil in [False, True]:
        blimp_text.USE_PIL = use_pil

        # ~~~~ standard text ~~~~

        print()
        print(f"standard_text use_pil={use_pil}")
        print("----------")

        # get the text image using text_standard

        im_res = blimp.text_standard(
            text, (0, 0), font, text_color, text_stroke_width)

        # extend the border and add guides

        im_res_expanded = trim.expand_border(np.array(im_res), border_size, border_size)
        # add_guides(im_res_expanded, im_res.size)
        im_res_expanded = Image.fromarray(im_res_expanded)

        # get the same image with borders and add guides

        im_res_border = blimp.text_standard(
            text, (border_size, border_size), font, text_color,
            text_stroke_width)
        im_res_border = np.array(im_res_border)
        # add_guides(im_res_border, im_res.size)
        im_res_border = Image.fromarray(im_res_border)

        # ~~~~ custom kerning text ~~~~

        print()
        print(f"custom_kerning use_pil={use_pil}")
        print("----------")

        # get the text image using text_standard

        im_custom = blimp.text_custom_kerning(
            text, (0, 0), font, text_color,
            text_stroke_width, text_kern_add, debug_lines)

        # extend the border and add guides

        im_custom_expanded = trim.expand_border(np.array(im_custom), border_size, border_size)
        # add_guides(im_custom_expanded, im_custom.size)
        im_custom_expanded = Image.fromarray(im_custom_expanded)

        # get the same image with borders and add guides

        im_custom_border = blimp.text_custom_kerning(
            text, (border_size, border_size), font, text_color,
            text_stroke_width, text_kern_add, debug_lines)
        im_custom_border = np.array(im_custom_border)
        # add_guides(im_custom_border, im_custom.size)
        im_custom_border = Image.fromarray(im_custom_border)

        # verify that sizes match

        assert im_res_border.size == im_res_expanded.size
        assert im_custom_border.size == im_custom_expanded.size

        # verify that nothing lies beyond the border for the expanded version
        # (we could test this with custom kerning as well if we disabled debug lines)

        start_x, end_x = trim.find_trim_x_indices(np.array(im_res_expanded))
        assert start_x >= border_size
        assert end_x <= im_res_expanded.size[0] - border_size

        if font.getname() == ("Cinzel", "Regular"):
            # serifs lie outside border
            start_x, end_x = trim.find_trim_x_indices(np.array(im_res_border))
            assert start_x < border_size
            if not use_pil:
                assert end_x > im_res_expanded.size[0] - border_size

        if not debug_lines:
            # verify expected colors
            # Note that PIL right now does not do colors properly!
            for img in [im_res_border, im_res_expanded, im_custom_border, im_res_expanded]:
                img_array = np.array(img)
                print("r:", set(np.unique(img_array[:, :, 0])))
                print("g:", set(np.unique(img_array[:, :, 1])))
                print("b:", set(np.unique(img_array[:, :, 2])))
                print("a count:", len(set(np.unique(img_array[:, :, 3]))))
                print()
                assert np.alltrue(img_array[:, :, 0] == 0)
                assert set(np.unique(img_array[:, :, 1])) == {0, 255}
                assert set(np.unique(img_array[:, :, 2])) == {0, 255}
                assert len(set(np.unique(img_array[:, :, 3]))) > 2
                assert np.alltrue(img_array[:, :, 3] <= 128)

        # ~~~~ save stuff ~~~~

        method_str = "scala" if not use_pil else "pil"

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
