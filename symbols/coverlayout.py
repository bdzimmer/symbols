"""

Functions for paperback cover layout images.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from typing import Tuple

import attr
import numpy as np

# constants for KDP, in inches

KDP_BLEED = 0.125
KDP_CREAM_PAPER_THICKNESS = 0.0025
KDP_WHITE_PAPER_THICKNESS = 0.002252
KDP_COLOR_PAPER_THICKNESS = 0.002347
KDP_SPINE_TOLERANCE = 0.0625


@attr.s(frozen=True)
class CoverLayoutInfo:
    """cover layout info"""

    cover_width = attr.ib()
    cover_height = attr.ib()
    spine_width = attr.ib()

    # horizontal locations of vertical lines
    bleed_v_left = attr.ib()
    bleed_v_right = attr.ib()
    spine_v_left = attr.ib()
    spine_v_right = attr.ib()
    spine_text_v_left = attr.ib()
    spine_text_v_right = attr.ib()

    # vertical locations (top -> y = 0) of horizontal lines
    bleed_h_top = attr.ib()
    bleed_h_bottom = attr.ib()

    verticals = attr.ib()
    horizontals = attr.ib()


def kdp_cover_dimensions(trim_width, trim_height, page_count, bleed, paper_thickness):
    """given trim width, page count, and paper thickness (all in inches),
    derive various KDP cover dimensions."""

    spine_width = paper_thickness * page_count
    cover_width = 2.0 * (bleed + trim_width) + spine_width
    cover_height = 2.0 * bleed + trim_height

    return cover_width, cover_height, spine_width


def kdp_cover_layout_info(
        trim_width,
        trim_height,
        page_count,
        bleed,
        paper_thickness,
        spine_tolerance):

    """calculate cover layout info"""

    cover_width, cover_height, spine_width = kdp_cover_dimensions(
        trim_width, trim_height, page_count, bleed, paper_thickness)

    # horizontal locations of vertical lines
    bleed_v_left = bleed
    bleed_v_right = cover_width - bleed
    spine_v_left = bleed + trim_width
    spine_v_right = cover_width - bleed - trim_width
    spine_text_v_left = spine_v_left + spine_tolerance
    spine_text_v_right = spine_v_right - spine_tolerance

    # vertical locations (top -> y = 0) of horizontal lines
    bleed_h_top = bleed
    bleed_h_bottom = cover_height - bleed

    verticals = [
        bleed_v_left,
        bleed_v_right,
        spine_v_left,
        spine_v_right,
        spine_text_v_left,
        spine_text_v_right
    ]

    horizontals = [
        bleed_h_top,
        bleed_h_bottom
    ]

    info = CoverLayoutInfo(
        cover_width=cover_width,
        cover_height=cover_height,
        spine_width=spine_width,
        bleed_v_left=bleed_v_left,
        bleed_v_right=bleed_v_right,
        spine_v_left=spine_v_left,
        spine_v_right=spine_v_right,
        spine_text_v_left=spine_text_v_left,
        spine_text_v_right=spine_text_v_right,
        bleed_h_top=bleed_h_top,
        bleed_h_bottom=bleed_h_bottom,
        verticals=verticals,
        horizontals=horizontals
    )

    return info


def template_image(info: CoverLayoutInfo, dpi: int, color: Tuple) -> np.ndarray:
    """get a cover layout template image"""

    canvas_width = int(info.cover_width * dpi)
    canvas_height = int(info.cover_height * dpi)

    img = np.zeros((canvas_height, canvas_width, 4), dtype=np.uint8)

    def draw_v(x_pos):
        """draw a 3px vertical line at x"""
        img[:, x_pos, :] = color
        img[:, x_pos - 1, :] = color
        img[:, x_pos + 1, :] = color

    def draw_h(y_pos):
        """draw a 3px horizontal line at y"""
        img[y_pos, :, :] = color
        img[y_pos - 1, :, :] = color
        img[y_pos + 1, :, :] = color

    for vert in info.verticals:
        draw_v(int(vert * dpi))

    for horiz in info.horizontals:
        draw_h(int(horiz * dpi))

    return img
