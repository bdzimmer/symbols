"""

Tests for particles.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import os
from typing import Tuple

import numpy as np
from PIL import Image

from symbols import particles, composite
from symbols import draw_cairo, symbols, transforms


DEBUG = True
SCRATCH_DIRNAME = os.path.join("test_scratch", "particles")


def test_gaussian():
    """akdsjfladkjfh lajkdf"""

    canvas_x = 1920
    canvas_y = 1080
    n_particles = 4000
    particle_size = 8
    particle_color = (32, 255, 255, 255)
    blur_size = 63
    blur_strength = 3.0

    blend_funcs = {
        "alpha": composite.alpha_blend,
        "additive": composite.additive_blend
    }

    # sample a ton of particles
    np.random.seed(1)
    p_xys = np.random.randn(n_particles, 2)
    p_xys = p_xys * np.array([canvas_x * 0.1, canvas_y * 0.1])[np.newaxis, :]
    canvas_center = np.array([canvas_x * 0.5, canvas_y * 0.5])
    p_xys = p_xys + canvas_center[np.newaxis, :]
    p_xys = np.array(p_xys, np.int)

    # build particle cache / drawing function

    tex_cache = {}

    def exp_particle_draw(pos, size, color, blend_func):
        """helper"""
        key = (size, color)
        if key in tex_cache:
            tex = tex_cache[key]
        else:
            tex = particles.default_glow_texture(color, size, blur_size, blur_strength)
            tex_cache[key] = tex

        particles.draw_texture(
            canvas,
            (pos[0] - tex.shape[1] // 2, pos[1] - tex.shape[0] // 2),
            tex,
            blend_func)

    # render all particles
    for blend_method_name in ["alpha", "additive"]:
        canvas = np.zeros((canvas_y, canvas_x, 4), dtype=np.uint8)
        blend_func = blend_funcs[blend_method_name]
        for idx in range(p_xys.shape[0]):
            if (idx % 1000) == 0:
                print("\t", idx + 1, "/", p_xys.shape[0])
            p_xy = p_xys[idx, :]
            age = 1.0 - np.linalg.norm(p_xy - canvas_center) / canvas_y
            age = round(age * 100.0) / 100.0
            color = _color_func(age, 255)
            exp_particle_draw(p_xy, particle_size, color, blend_func)

        # TODO: assertions

        _debug_save_image(Image.fromarray(canvas), f"particles_{blend_method_name}.png")

    print("texture cache size:", len(tex_cache))


def test_blend():
    """test blending functions"""

    canvas_x = 1280
    canvas_y = 1280
    center = (canvas_x // 2, canvas_y // 2)

    blend_funcs = {
        "alpha": composite.alpha_blend,
        "additive": composite.additive_blend
    }

    triangle_points = [
        symbols.add(x, center)
        for x in transforms.points_around_circle(3, symbols.TAU * 0.25, 200)]

    alpha = 128
    dots = [
        _dot_image(800, x)
        for x in [(255, 0, 0, alpha), (0, 255, 0, alpha), (0, 0, 255, alpha)]]

    for blend_method_name in ["alpha", "additive"]:
        canvas = np.zeros((canvas_y, canvas_x, 4), dtype=np.uint8)
        blend_func = blend_funcs[blend_method_name]
        for tri_pt, dot in zip(triangle_points, dots):
            c_x, c_y = symbols.add((int(tri_pt[0]), int(tri_pt[1])), (-400, -400))
            res = blend_func(dot, canvas[c_y:(c_y + 800), c_x:(c_x + 800), :])
            print(
                blend_method_name, "\n",
                "\tdot:", np.min(dot[:, :, 3]), np.max(dot[:, :, 3]), "\n",
                "\tdst rgb:", np.min(canvas[c_y:(c_y + 800), c_x:(c_x + 800), 0:3]),
                np.max(canvas[c_y:(c_y + 800), c_x:(c_x + 800), 0:3]), "\n",
                "\tdst a:", np.min(canvas[c_y:(c_y + 800), c_x:(c_x + 800), 3]),
                np.max(canvas[c_y:(c_y + 800), c_x:(c_x + 800), 3]), "\n",
                "\tres a:", np.min(res[:, :, 3]), np.max(res[:, :, 3])),
            canvas[c_y:(c_y + 800), c_x:(c_x + 800), :] = res

        # TODO: assertions
        _debug_save_image(Image.fromarray(canvas), f"dots_{blend_method_name}.png")


def _color_func(age, alpha):
    """default color function"""
    age = max(min(age, 1.0), 0.0)
    return (
        int(32 * age * age),  # nonzero red allows saturation to white
        int(255 * age * age),
        int(255 * age),
        alpha * age * age  # 255
    )


def _dot_image(diameter: float, color: Tuple) -> np.ndarray:
    """get an image of a dot"""
    surface, context = draw_cairo.create_surface(diameter, diameter)
    draw_cairo.draw_dot(
        context,
        symbols.Dot((diameter / 2, diameter / 2), diameter / 2, 0, 0, color))
    dot_image = draw_cairo.surface_to_image(surface)
    return np.array(dot_image)


def _debug_save_image(img: Image, filename: str):
    """save an image"""
    if DEBUG:
        os.makedirs(SCRATCH_DIRNAME, exist_ok=True)
        output_filename = os.path.join(SCRATCH_DIRNAME, filename)
        print(output_filename)
        img.save(output_filename)
