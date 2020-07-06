"""

Utilities for drawing text, using an external executable.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import hashlib
from typing import Tuple, Any, Dict, Union
import os

import cv2
import numpy as np
from PIL import Image

from symbols import conversions


Font = Tuple[str, str, int]
ColorRGB = Tuple[int, int, int]
ColorRGBA = Tuple[int, int, int, int]
Color = Union[ColorRGB, ColorRGBA]

IMAGE_DIRNAME = "text_cache"
BIN_DIRNAME = [
    "C:/Projects/code/secondary/dist",
    "/home/ben/code/secondary/dist"][1]


def get_info(
        text: str,
        font: Font,
        stroke_width: int,
        border_size: Tuple[int, int]) -> Dict:
    """draw text using external tool"""

    print(f"info for '{text}'...", end="", flush=True)

    id_object = (text, font, stroke_width, border_size)
    id_string = compute_hash(id_object)

    info_filename = os.path.join(IMAGE_DIRNAME, "text_" + id_string + "_info.txt")
    if not os.path.exists(info_filename):
        config_filename = os.path.join(IMAGE_DIRNAME, "text_" + id_string + ".txt")
        os.makedirs(IMAGE_DIRNAME, exist_ok=True)
        _write_config(
            config_filename,
            text, font, stroke_width, border_size)
        command = _info_command(config_filename)
        os.system(command)

    info = _read_info(info_filename)

    print("done")

    return info


def draw(
        text: str,
        font: Font,
        stroke_width: int,
        border_size: Tuple[int, int]) -> Tuple[np.array, Dict]:
    """draw text using external tool"""

    id_object = (text, font, stroke_width, border_size)
    id_string = compute_hash(id_object)

    image_filename = os.path.join(IMAGE_DIRNAME, "text_" + id_string + ".png")
    if not os.path.exists(image_filename):
        config_filename = os.path.join(IMAGE_DIRNAME, "text_" + id_string + ".txt")
        os.makedirs(IMAGE_DIRNAME, exist_ok=True)
        _write_config(
            config_filename,
            text, font, stroke_width, border_size)
        command = _draw_command(config_filename)
        os.system(command)

    img = cv2.imread(image_filename, cv2.IMREAD_UNCHANGED)

    info_filename = os.path.join(IMAGE_DIRNAME, "text_" + id_string + "_info.txt")
    info = _read_info(info_filename)

    return img, info


def draw_on_image(
        im_dest: Image.Image,
        pos: Tuple[int, int],
        text: str,
        font: Font,
        fill: Color,
        stroke_width: int,
        border_size: Tuple[int, int]
        ) -> Dict[str, int]:

    """draw text onto a PIL Image, mutating it"""

    # pylint: disable=too-many-arguments

    # Assumes the destination is transparent, since text will be drawn onto it
    # via a simple masked paste.

    # Eventually, I'd like to get away from using PIL.
    # But this is the most convenient for testing at the moment.

    # Currently assumes that the PIL Image has an alpha channel.

    im_text, info = draw(text, font, stroke_width, border_size)
    im_text_alpha = im_text[:, :, 3]

    # adjust position to compensate for border_size
    pos_adj = (pos[0] - border_size[0], pos[1] - border_size[1])

    # colorize
    im_text = conversions.colorize(im_text_alpha, fill)

    # I think the only place where this will screw up is sequentially
    # drawn overlapping letters.
    # TODO: could possibly require a maxing operation for mask locations.
    im_text_pil = Image.fromarray(im_text)
    im_mask_pil = Image.fromarray(im_text[:, :, 3] > 0)
    im_dest.paste(im_text_pil, pos_adj, mask=im_mask_pil)

    return info


def compute_hash(val: Any) -> str:
    """compute a stable hash using a string representation of an object"""
    hasher = hashlib.sha256()
    hasher.update(str(val).encode("utf-8"))
    return hasher.hexdigest()


def _info_command(config_filename: str) -> str:
    """create the command to run the text executable"""
    jar_filename = os.path.join(BIN_DIRNAME, "secondary.jar")
    class_name = "bdzimmer.orbits.Text"
    return f"java -cp {jar_filename} {class_name} {config_filename} info"


def _draw_command(config_filename: str) -> str:
    """create the command to run the text executable"""
    jar_filename = os.path.join(BIN_DIRNAME, "secondary.jar")
    class_name = "bdzimmer.orbits.Text"
    return f"java -cp {jar_filename} {class_name} {config_filename} draw"


def _write_config(
        config_filename: str,
        text: str,
        font: Font,
        stroke_width: int,
        border_size: Tuple[int, int],
        ) -> None:
    """write config file"""

    with open(config_filename, "w") as config_file:
        config_file.write(text + "\n")
        name, style, size = font
        config_file.write(f"{name};{style};{size}\n")
        border_x, border_y = border_size
        config_file.write(f"{border_x};{border_y}\n")
        if stroke_width > 0:
            config_file.write(f"{stroke_width}\n")


def _read_info(info_filename: str) -> Dict[str, Any]:
    """read info into dictionary"""
    with open(info_filename, "r") as info_file:
        lines = info_file.readlines()
        info = {}
        for line in lines:
            key, val = line.strip().split("\t")
            if key == "stroke":
                if val is not None:
                    info[key] = float(val)
            else:
                info[key] = int(val)
    return info
