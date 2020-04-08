"""

Utilities for drawing text, using an external executable.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import hashlib
from typing import Tuple, Any
import re
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

Font = Tuple[str, str, int]

IMAGE_DIRNAME = "text_cache"
BIN_DIRNAME = "C:/Ben/code/secondary/dist"


def draw(text: str, font: Font) -> np.array:
    """draw text using external tool"""
    id_object = (str, font)
    id_string = compute_hash(id_object)

    image_filename = os.path.join(IMAGE_DIRNAME, "text_" + id_string + ".png")
    if not os.path.exists(image_filename):
        config_filename = os.path.join(IMAGE_DIRNAME, "text_" + id_string + ".txt")
        os.makedirs(IMAGE_DIRNAME, exist_ok=True)
        with open(config_filename, "w") as config_file:
            config_file.write(text + "\n")
            name, style, size = font
            config_file.write(f"{name};{style};{size}\n")
        command = draw_command(config_filename)
        os.system(command)

    im = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED)

    info_filename = os.path.join(IMAGE_DIRNAME, "text_" + id_string + "_info.txt")
    with open(info_filename, "r") as info_file:
        lines = info_file.readlines()
        info = {}
        for line in lines:
            key, val = line.split("\t")
            info[key] = int(val)

    return im, info


def compute_hash(val: Any) -> str:
    """compute a stable hash using a string representation of an object"""
    hasher = hashlib.sha256()
    hasher.update(str(val).encode("utf-8"))
    return hasher.hexdigest()


def draw_command(config_filename):
    """create the command to run the text executable"""
    jar_filename = os.path.join(BIN_DIRNAME, "secondary.jar")
    class_name = "bdzimmer.orbits.Text"
    return f"java -cp {jar_filename} {class_name} {config_filename}"


def size(text, font):
    """get the width and height (in pixels) of a string"""
    return font.getsize(text)


def offset(text, font):
    """get offset"""
    return font.getoffset(text)


def font_line_height(font):
    """get line height"""
    ascent, descent = font.getmetrics()
    return ascent + descent