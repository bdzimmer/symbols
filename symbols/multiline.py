"""

Utilities for drawing and animating multiline text.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import re
from typing import List, Any, Callable, Tuple

import numpy as np
from PIL import Image

from symbols import blimp_text, trim


def wrap_text(
        text: str,
        font: Any,
        width_max: int) -> List[str]:
    """split text into lines / words given a width in pixels"""

    # split into words
    words = re.split("\\s+", text)
    lines = []
    start_idx = 0

    for idx in range(1, len(words)):
        line = " ".join(words[start_idx:(idx + 1)])
        width, _ = blimp_text.getsize(font, line)  # font.getsize(line)

        if width > width_max:
            line_new = " ".join(words[start_idx:idx])
            lines.append(line_new)
            start_idx = idx

    line_new = " ".join(words[start_idx:])
    lines.append(line_new)

    return lines


def multiline(
        lines: List[str],
        font: Any,
        color: Tuple,
        line_height: int,
        box_xy: Tuple[int, int],
        border_xy: Tuple[int, int],
        justify_method: str) -> np.ndarray:
    """draw multiline text using blimp_text"""

    # pylint: disable=too-many-arguments

    # original behavior
    # image = Image.new("L", (image_width, image_height), 0)
    # draw = ImageDraw.Draw(image)
    #     draw.text((pos_x, pos_y), line, font=font, fill="white")

    box_x, box_y = box_xy
    border_x, border_y = border_xy
    image_width = box_x + border_x * 2
    image_height = box_y + border_y * 2

    image = Image.new("RGBA", (image_width, image_height), (0, 0, 0, 0))

    for idx, line, in enumerate(lines):
        print(line)
        pos_x = border_x
        pos_y = border_y + idx * line_height

        if (
                justify_method != "none" and
                (idx < len(lines) - 1) and
                lines[idx + 1] != " "
                and line != " "
        ):
            image_line = line_justified(line, font, color, line_height, box_xy, border_xy, justify_method)
            mask = Image.fromarray(np.array(image_line)[:, :, 3] > 0)
            image.paste(image_line, (0, idx * line_height), mask)
        else:
            blimp_text.text(image, (pos_x, pos_y), line, font, color, 0)

    return np.array(image)


def line_justified(
        line: str,
        font: Any,
        color: Tuple,
        line_height: int,
        box_xy: Tuple[int, int],
        border_xy: Tuple[int, int],
        justify_method: str
        ) -> Image.Image:
    """justified text"""

    # TODO: there are probably issues here for edge cases like no spaces, etc

    box_x, box_y = box_xy
    border_x, border_y = border_xy
    image_width = box_x + border_x * 2

    image_line = Image.new("RGBA", (image_width, line_height + border_y * 2), (0, 0, 0, 0))
    blimp_text.text(image_line, (border_x, border_y), line, font, color, 0)

    if justify_method == "trim":
        print("trim", image_line.size, "->", end="")
        image_line = _trim_and_expand(np.array(image_line), border_x)
        image_line = Image.fromarray(image_line)
        print(image_line.size)

    # find the locations of the spaces
    image_line_np = np.array(image_line)
    image_line_zero = ~np.any(image_line_np[:, :, 3] > 0, axis=0)
    space_locations = []
    for char_idx, char in enumerate(line):
        if char == " ":
            line_before_space = line[0:char_idx]
            if justify_method == "trim":
                image_line_before_space = Image.new("RGBA", (image_width, line_height + border_y * 2), (0, 0, 0, 0))
                blimp_text.text(image_line_before_space, (border_x, border_y), line_before_space, font, color, 0)
                trim_idxs = trim.find_trim_x_indices(np.array(image_line_before_space))
                width_before_space = trim_idxs[1] - trim_idxs[0]
            else:
                width_before_space = blimp_text.getsize(font, line_before_space)[0]
            # find the first alpha 0 column after width_befeore_spaces
            # TODO: logic for if no insert point???
            insert_idx = np.where(image_line_zero[(border_x + width_before_space):])[0][0]
            insert_idx = border_x + width_before_space + insert_idx
            space_locations.append(insert_idx)

    if len(space_locations) > 0:
        # find remainder of width
        if justify_method == "trim":
            trim_idxs = trim.find_trim_x_indices(np.array(image_line))
            width_line = trim_idxs[1] - trim_idxs[0]
        else:
            width_line = blimp_text.getsize(font, line)[0]
        width_extra = box_x - width_line
        # divide between insert locations
        num_even, num_extra = divmod(width_extra, len(space_locations))
        insert_amounts = np.ones(len(space_locations)) * num_even
        insert_amounts[:num_extra] += 1
        insert_amounts = np.array(insert_amounts, dtype=np.int)

        assert np.sum(insert_amounts) == width_extra

        image_line_new = Image.new(
            "RGBA", (image_width, line_height + border_y * 2), (0, 0, 0, 0))
        image_line_new_np = np.array(image_line_new)

        # get each
        read_chunk_start = 0
        write_chunk_start = 0
        for space_location, insert_amount in zip(space_locations, insert_amounts):
            chunk_width = space_location - read_chunk_start
            image_line_new_np[:, write_chunk_start:(write_chunk_start + chunk_width), :] = (
                image_line_np[:, read_chunk_start:(read_chunk_start + chunk_width), :]
            )
            read_chunk_start = space_location
            write_chunk_start = write_chunk_start + chunk_width + insert_amount

        # get the last word
        chunk_width = image_width - write_chunk_start
        image_line_new_np[:, write_chunk_start:(write_chunk_start + chunk_width), :] = (
            image_line_np[:, read_chunk_start:(read_chunk_start + chunk_width), :]
        )

        image_line_new = Image.fromarray(image_line_new_np)
        image_line = image_line_new

    return image_line


def _trim_and_expand(image_line: np.ndarray, border_x: int) -> np.ndarray:
    """trim and restore borders of a line of text"""
    start_x, end_x = trim.find_trim_x_indices(image_line)
    image = image_line[:, start_x:end_x, :]
    image_line = trim.expand_border(image, border_x, 0)
    return image_line


def animate_characters(
        lines: List[str],
        font: Any,
        color: Tuple,
        width_max: int,
        border_xy: Tuple[int, int],
        justify: bool,
        im_func: Callable,     # function to update the image before writing to disk
        frame_func: Callable,  # function to write frame to disk
        dup: int,              # frames per character
        dup_end: int           # duplicate frames at end
        ) -> None:
    """animate text by individual characters"""

    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    line_lengths = [len(x) for x in lines]
    length_all = sum(line_lengths)
    # line_height = font_line_height(font)
    ascent, descent = blimp_text.getmetrics(font)
    line_height = ascent + descent
    # TODO: potentially add leading to line_height
    image_height = line_height * len(lines)

    idx_frame_out = 0

    for idx_frame in range(length_all):

        print(idx_frame + 1, "/", length_all)

        # calculate lines_mod
        # TODO: other animation modes
        # TODO: optionally include extra "characters" for line ends
        lines_mod = []
        chars_total = 0
        for idx, line in enumerate(lines):
            if idx_frame >= chars_total + line_lengths[idx]:
                lines_mod.append(line)
                chars_total = chars_total + len(line)
            else:
                partial = line[0:(idx_frame - chars_total + 1)]
                lines_mod.append(partial)
                print(partial)
                break

        # TODO: test a case where one word is too long
        img = multiline(
            lines_mod, font, color, line_height, (width_max, image_height), border_xy, justify)

        for _ in range(dup):
            im_mod = im_func(img)
            frame_func(im_mod)
            idx_frame_out = idx_frame_out + 1

    for _ in range(dup_end):
        im_mod = im_func(img)
        frame_func(im_mod)
        idx_frame_out = idx_frame_out + 1
