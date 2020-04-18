"""

Test text functions.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import unittest

import cv2
from symbols import text_scala as text_scala

from PIL import Image


class TestsTextScala(unittest.TestCase):
    """Tests for text_scala"""

    def test_draw(self):
        """test draw"""
        font = ("Cinzel", "plain", 64)
        im, info = text_scala.draw("AVIARY", font, 0, (32, 32))
        print(info)
        print(im.shape)
        cv2.imwrite("text_scala_0.png", im)

    def test_draw_on_image(self):
        """test draw_on_image"""
        font = ("Cinzel", "plain", 64)
        im = Image.new("RGBA", (640, 480), (0, 0, 0))
        text_scala.draw_on_image(
            im, (64, 64), "AVIARY", font, (0, 0, 255), 1, (32, 32))
        im.save("text_scala_1.png")
