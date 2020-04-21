"""

Utilities for drawing text, using an external executable.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import hashlib
from typing import Tuple, Any, Dict
import os

import cv2
import numpy as np
from PIL import Image

Font = Tuple[str, str, int]
Color = Tuple[int, int, int]

IMAGE_DIRNAME = "text_cache"
BIN_DIRNAME = [
    "C:/Projects/code/secondary/dist",
    "/home/ben/code/secondary/dist"][1]


def draw(
        text: str,
        font: Font,
        stroke_width: int,
        border_size: Tuple[int, int]) -> np.array:
    """draw text using external tool"""

    print(stroke_width, border_size)

    id_object = (text, font, stroke_width, border_size)
    id_string = compute_hash(id_object)

    image_filename = os.path.join(IMAGE_DIRNAME, "text_" + id_string + ".png")
    if not os.path.exists(image_filename):
        config_filename = os.path.join(IMAGE_DIRNAME, "text_" + id_string + ".txt")
        os.makedirs(IMAGE_DIRNAME, exist_ok=True)
        with open(config_filename, "w") as config_file:
            config_file.write(text + "\n")
            name, style, size = font
            config_file.write(f"{name};{style};{size}\n")
            border_x, border_y = border_size
            config_file.write(f"{border_x};{border_y}\n")
            if stroke_width > 0:
                config_file.write(f"{stroke_width}\n")
        command = draw_command(config_filename)
        os.system(command)

    im = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED)

    info_filename = os.path.join(IMAGE_DIRNAME, "text_" + id_string + "_info.txt")
    with open(info_filename, "r") as info_file:
        lines = info_file.readlines()
        info = {}
        for line in lines:
            key, val = line.strip().split("\t")
            if key == "stroke":
                if val is not None:
                    info[key] = float(stroke_width)
            else:
                info[key] = int(val)

    return im, info


def draw_on_image(
        im: Image,
        pos: Tuple[int, int],
        text: str,
        font: Font,
        fill: Color,
        stroke_width: int,
        border_size: Tuple[int, int]
        ) -> Dict[str, int]:

    """draw text onto a PIL Image, mutating it"""

    # Eventually, I'd like to get away from using PIL.
    # But this is the most convenient for testing at the moment.

    # Currently assumes that the PIL Image has an alpha channel.

    im_text, info = draw(text, font, stroke_width, border_size)

    # adjust position with border_size
    pos_adj = (pos[0] - border_size[0], pos[1] - border_size[1])

    # colorize
    im_text = colorize(im_text, fill)

    # alpha blend using PIL
    if len(im.size) > 2:
        im_text_pil = Image.fromarray(im_text)
        im.alpha_composite(im_text_pil, pos_adj)
    else:
        im_text_pil = Image.fromarray(im_text[:, :, 3])
        im_mask_pil = Image.fromarray(im_text[:, :, 3] > 0)
        im.paste(im_text_pil, pos_adj, mask=im_mask_pil)

    return info


def colorize(im: np.array, color: Color) -> np.array:
    """colorize a white / alpha image"""
    im = np.copy(im)
    # colorize everyting
    im[:, :, 0:3] = color
    # set all completely transparent pixels to (something, 0)
    im[im[:, :, 3] == 0, 0:3] = (0, 0, 0)
    return im


def compute_hash(val: Any) -> str:
    """compute a stable hash using a string representation of an object"""
    hasher = hashlib.sha256()
    hasher.update(str(val).encode("utf-8"))
    return hasher.hexdigest()


def draw_command(config_filename: str) -> str:
    """create the command to run the text executable"""
    jar_filename = os.path.join(BIN_DIRNAME, "secondary.jar")
    class_name = "bdzimmer.orbits.Text"
    return f"java -cp {jar_filename} {class_name} {config_filename}"
