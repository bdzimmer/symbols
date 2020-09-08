"""
Generate 3D cover images.
"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from symbols import blimp, util, symbols, trim
from symbols import hopf, transforms
from symbols import coverlayout as cl


INTERP_MODE = cv2.INTER_LANCZOS4


def composite_image(
        canvas_size: Tuple[int, int],
        trim_size: Tuple[float, float],
        info: cl.CoverLayoutInfo,
        wrap_image: np.ndarray,
        ebook_cover_image: np.ndarray,
        dpi: int,
        shift_xy: Tuple[int, int],
        bg_multiply: float
        ) -> Tuple[np.ndarray, np.ndarray]:
    """a higher-level function for making an artsy cover composite image"""

    # includes lots of defaults for cover_image_3d

    cover_img = cover_image_3d(
        canvas_size=canvas_size,
        trim_size=trim_size,
        info=info,
        wrap_image=wrap_image,
        dpi=dpi,
        y_angle=symbols.TAU * 0.09,
        view_pos_scale=1.25,
        view_shift=10.0,
        spine_shade=0.8,
        wireframe=False
    )

    trim_x = trim.find_trim_x_indices(cover_img)
    trim_y = trim.find_trim_y_indices(cover_img)
    cover_img = cover_img[:, trim_x[0]:trim_x[1], :]
    cover_img = cover_img[trim_y[0]:trim_y[1], :]

    cover_img_trimmed, shift_xy = trim.trim(
        cover_img, shift_xy, canvas_size)

    cover_img_pil = Image.fromarray(cover_img_trimmed)
    img_bg, _ = trim.trim(
        ebook_cover_image,
        (int(0.5 * (canvas_size[0] - ebook_cover_image.shape[1])),
         int(0.5 * (canvas_size[1] - ebook_cover_image.shape[0]))),
        canvas_size)

    img_bg = img_bg[:, :, 0:3]
    img_bg = np.array(img_bg * bg_multiply, dtype=np.ubyte)
    img_bg = blimp.add_alpha(img_bg)
    img_bg_pil = Image.fromarray(img_bg)
    img_bg_pil.alpha_composite(cover_img_pil, shift_xy)

    return np.array(img_bg_pil), cover_img


def cover_image_3d(
        canvas_size: Tuple[int, int],
        trim_size: Tuple[float, float],
        info: cl.CoverLayoutInfo,
        wrap_image: np.ndarray,
        dpi: int,
        wireframe: bool,
        y_angle: float,         # rotation around vertical axis (symbols.TAU * 0.1)
        view_pos_scale: float,  # camera distance as multiple of trim_height (1.25)
        view_shift: float,      # larger number for less perspective (10.0)
        spine_shade: float      # shading factor for spine (0.8)
        ) -> np.ndarray:

    """generate a 3D cover image from info and wrap image"""

    trim_width, trim_height = trim_size
    canvas_width, canvas_height = canvas_size

    canvas = np.zeros((canvas_height, canvas_width, 4), dtype=np.ubyte)

    cover_shift = np.array([0.5 * trim_width, 0.5 * trim_height, 0.0])[np.newaxis, :]

    # top right, top left, bottom left, bottom right

    cover_points_2d = np.array([
        [info.spine_v_right, info.bleed_h_top],
        [info.bleed_v_right, info.bleed_h_top],
        [info.bleed_v_right, info.bleed_h_bottom],
        [info.spine_v_right, info.bleed_h_bottom]
    ]) * dpi

    cover_points_3d = np.array([
        [0.0, 0.0, 0.0],
        [trim_width, 0.0, 0.0],
        [trim_width, trim_height, 0.0],
        [0.0, trim_height, 0.0]
    ]) - cover_shift

    spine_points_2d = np.array([
        [info.spine_v_left, info.bleed_h_top],
        [info.spine_v_right, info.bleed_h_top],
        [info.spine_v_right, info.bleed_h_bottom],
        [info.spine_v_left, info.bleed_h_bottom]
    ]) * dpi

    spine_points_3d = np.array([
        [0.0, 0.0, -info.spine_width],
        [0.0, 0.0, 0.0],
        [0.0, trim_height, 0.0],
        [0.0, trim_height, -info.spine_width]
    ]) - cover_shift

    scale_factor = canvas_height / trim_height
    view_pos = np.array([0.0, 0.0, scale_factor * view_shift])  # 0 0 800

    # looking down at origin from a height of cover_height
    cam_z_pos = view_pos_scale * view_shift
    cam_pos = np.array([0.0, 0.0, cam_z_pos])

    cam_trans = np.dot(
        util.rotation_z(0.5 * symbols.TAU),
        transforms.transformation(np.identity(3), -cam_pos))

    # add an aditional y rotation
    cam_trans = np.dot(cam_trans, util.rotation_y(y_angle))

    for (pts_type, pts_2d, pts_3d) in [
            ("cover", cover_points_2d, cover_points_3d),
            ("spine", spine_points_2d, spine_points_3d)]:

        pts_p = np.transpose(
            hopf.apply_perpsective_transformation(
                np.transpose(pts_3d),
                cam_trans, view_pos, canvas_size))

        # assert pts_p.shape == (4, 2)
        # assert pts_2d.shape == (4, 2)

        # OpenCV doesn't understand the y-axis flip that we are doing
        # in our perspective transformation calculations
        pts_2d_reordered = pts_2d[[3, 2, 1, 0], :]

        perspective = cv2.getPerspectiveTransform(
            pts_2d_reordered.astype(np.float32),
            pts_p.astype(np.float32))

        if wireframe:
            cv2.polylines(
                canvas, [pts_p.astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
        else:
            warped = cv2.warpPerspective(
                wrap_image,
                perspective,
                canvas_size,
                flags=INTERP_MODE)
            mask = np.zeros((canvas_height, canvas_width), dtype=np.ubyte)
            cv2.fillConvexPoly(mask, pts_p.astype(np.int32), 255, cv2.LINE_AA)
            if pts_type == "spine":
                warped = warped * spine_shade
            canvas[mask > 0, 0:3] = warped[mask > 0]
            canvas[mask > 0, 3] = np.maximum(canvas[mask > 0, 3], mask[mask > 0])

    return canvas
