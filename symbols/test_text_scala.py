"""

Test text functions.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import unittest

import cv2
from symbols import text_scala as text

from PIL import Image


class TestsTextScala(unittest.TestCase):
    """Tests for text_scala"""

    def test_draw(self):
        """test draw"""
        font = ("Cinzel", "plain", 64)
        im, info = text.draw("AVIARY", font)
        print(info)
        print(im.shape)
        cv2.imwrite("text_scala_0.png", im)

    def test_draw_on_image(self):
        """test draw_on_image"""
        font = ("Cinzel", "plain", 64)
        im = Image.new("RGBA", (640, 480), (0, 0, 0))
        text.draw_on_image(im, (64, 64), "AVIARY", font, (0, 0, 64))
        im.save("text_scala_1.png")
