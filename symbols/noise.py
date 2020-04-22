"""

Effects based on noise.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.


import numpy as np
from PIL import Image, ImageDraw

from symbols import transforms


def curl_field_scalar(x, y, z, noise):
    """calculate a 3D velocity field using scalar curl noise"""

    eps = 1.0e-4

    def deriv(a1, a2):
        return (a1 - a2) / (2.0 * eps)

    dx = deriv(noise(x + eps, y, z), noise(x - eps, y, z))
    dy = deriv(noise(x, y + eps, z), noise(x, y - eps, z))
    dz = deriv(noise(x, y, z + eps), noise(x, y, z - eps))

    return dy - dz, dz - dx, dx - dy


def curl_field_vector(x, y, z, px, py, pz):
    """calculate a 3D velocity field using vector curl noise"""

    eps = 1.0e-4
    offset = 100.0

    def deriv(a1, a2):
        return (a1 - a2) / (2.0 * eps)

    # x_dx = deriv(px(x + eps, y, z), px(x - eps, y, z))
    x_dy = deriv(px(x, y + eps, z), px(x, y - eps, z))
    x_dz = deriv(px(x, y, z + eps), px(x, y, z - eps))

    y_dx = deriv(py(offset + x + eps, y, z), py(offset + x - eps, y, z))
    # y_dy = deriv(py(offset + x, y + eps, z), py(offset + x, y - eps, z))
    y_dz = deriv(py(offset + x, y, z + eps), py(offset + x, y, z - eps))

    z_dx = deriv(pz(x + eps, offset + y, z), pz(x - eps, offset + y, z))
    z_dy = deriv(pz(x, offset + y + eps, z), pz(x, offset + y - eps, z))
    # z_dz = deriv(pz(x, offset + y, z + eps), pz(x, offset + y, z - eps))

    return z_dy - y_dz, x_dz - z_dx, y_dx - x_dy


def ramp(x):
    """smooth ramp function from curl noise paper"""
    if x > 1.0:
        return 1.0
    elif x < -1.0:
        return -1.0
    else:
        return 15.0 / 8.0 * x - 10.0 / 8.0 * x * x * x + 3.0 / 8.0 * x * x * x * x * x



def draw_frame(
        canvas,
        p_poss,     # list of particle positions at each timestep
        actives,    # list of which particles are active at each timestep
        cam_trans,  # camera transform
        view_pos,   # view position

        max_life,
        trail_length,
        particle_size,
        depth_sort,

        color_func,  # color given 0.0-1.0 age value
        build_draw_func,   # function for drawing on canvas

        prof        # profiler
        ):

    prof.tick("trans")

    coords, depths, age_tail, age_head = noise_geom(
        canvas,
        p_poss,
        actives,
        cam_trans,
        view_pos
    )

    prof.tock("trans")

    prof.tick("draw")

    canvas_draw, canvas_draw_get = build_draw_func(canvas)

    draw(
        coords, depths, age_tail, age_head,
        max_life,
        trail_length,
        particle_size,
        depth_sort,
        color_func,
        canvas_draw,
    )

    canvas = canvas_draw_get()

    return canvas


def noise_geom(
        canvas,
        p_poss,     # list of particle positions at each timestep
        actives,    # list of which particles are active at each timestep
        cam_trans,  # camera transform
        view_pos,   # view position
        ):

    """
    Draw a single frame of particles on a canvas image, modifying it in-place.
    """

    height, width, _ = canvas.shape
    p_shift = np.array([width * 0.5, height * 0.5])[:, np.newaxis]

    # TODO: only keep the last X for the trail length of the oldest particle?

    # list of arrays
    p_pos_ps = []

    for (p_pos, p_active) in p_poss:  # render all positions up to and including current position

        # apply camera and perspective transformations
        p_pos_c = transforms.transform(cam_trans, np.transpose(p_pos))
        p_pos_p = transforms.perspective(p_pos_c, view_pos)
        # align and convert to int
        # TODO: flip y properly
        p_pos_p[1, :] = 0.0 - p_pos_p[1, :]
        p_pos_p = np.clip(np.array(p_pos_p + p_shift), -width * 2, height * 2)
        p_pos_p = np.array(p_pos_p, dtype=np.int)
        p_pos_p = np.transpose(p_pos_p)

        # TODO: build a list of particles to render, with depths
        p_pos_ps.append((p_pos_p, p_pos_c[2, :]))  # smaller z is farther away

    # ~~~~ ~~~~ ~~~~ ~~~~

    # stack arrays

    coords = np.concatenate([x[0] for x in p_pos_ps], axis=0)
    depths = np.concatenate([x[1] for x in p_pos_ps], axis=0)
    actives_head = np.concatenate([p_poss[-1][1] for _ in range(len(p_poss))], axis=0)
    age_tail = actives        # age at current tail position?
    age_head = actives_head   # age at head in current frame?

    return coords, depths, age_tail, age_head


def draw(
        coords, age_tail, age_head,

        max_life,
        trail_length,
        particle_size,

        color_func,   # color given 0.0-1.0 age value
        canvas_draw,  # function for drawing on canvas

    ):

    """draw a bunch of particles"""

    for p_idx in range(coords.shape[0]):

        p_age_tail = age_tail[p_idx]
        p_age_head = age_head[p_idx]

        if -1 < p_age_tail <= max_life and p_age_head > -1:
            age_diff = p_age_head - p_age_tail  # age difference between head and tail
            color_scale = max(trail_length - age_diff, 0) / trail_length
            color = color_func(color_scale)
            x, y = coords[p_idx, :]
            canvas_draw((x, y), particle_size, color)


def ethereal_color_func(age):
    """default color function"""
    # TODO: this clamp / clip shouldn't be necessary
    age = max(min(age, 1.0), 0.0)
    return (
        int(255 * age),
        int(255 * age * age),
        0)


def build_pil_draw_func(canvas, composite):
    """draw a point on a canvas with OpenCV"""

    canvas_im = Image.fromarray(canvas, "RGBA")
    if composite:
        canvas_comp = Image.new("RGBA", canvas_im.size, (0, 0, 0, 0))
        canvas_draw = ImageDraw.Draw(canvas_comp)
    else:
        canvas_draw = ImageDraw.Draw(canvas_im)

    def draw(pos, particle_size, color):
        """do the drawing"""

        x, y = pos
        if particle_size == 0:
            canvas_draw.point([(x, y)], color)
        else:
            canvas_draw.ellipse(
                [x - particle_size,
                 y - particle_size,
                 x + particle_size,
                 y + particle_size],
                color)
        if composite:
            # canvas_im.alpha_composite(canvas_comp)
            crop_section = (
                [x - particle_size - 1,
                 y - particle_size - 1,
                 x + particle_size + 1,
                 y + particle_size + 1]
            )
            canvas_im.alpha_composite(
                canvas_comp.crop(crop_section),
                (x - particle_size - 1,
                 y - particle_size - 1)
            )
            canvas_comp.paste(
                (0, 0, 0, 0),
                # (0, 0, canvas_comp.size[0], canvas_comp.size[1])
                crop_section
            )

    def get():
        return np.array(canvas_im)

    return draw, get
