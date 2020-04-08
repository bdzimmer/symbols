"""

Test text functions.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import unittest

import cv2

from symbols import text_scala as text


class TestsTextScala(unittest.TestCase):
    """Tests for text_scala"""

    def test_draw(self):
        """test draw_text"""
        font = ("Cinzel", "plain", 64)
        im, info = text.draw("AVIARY", font)
        print(info)
        print(im.shape)
        cv2.imwrite("text_scala_0.png", im)
