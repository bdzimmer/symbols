"""

Test blimp_text

"""

# Copyright (c) 2020 Ben Zimmer. All rights resrved.

import os

from PIL import Image

from symbols import blimp, blimp_text, debugutil


DEBUG_VISUALIZE = os.environ.get("DEBUG_VISUALIZE") == "true"


def test_text():
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
            im_res = Image.new("RGBA", size, (0, 0, 0, 0))
            im_res = blimp.text_standard(*args)
            # blimp_text.text(im_res, (0, 0), *args)
            assert im_res.size == size
            assert im_res.mode == "RGBA"
            if DEBUG_VISUALIZE:
                debugutil.show(im_res, "test_text_standard " + str(use_pil))
            im_res.save("text_" + str(use_pil) + ".png")