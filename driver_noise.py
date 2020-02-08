"""

Driver for curl noise experimentation.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.


from datetime import datetime
import math
import os
import pickle
import random

import cv2
import numpy as np
from opensimplex import OpenSimplex
from PIL import Image, ImageDraw

from symbols import effects, util, transforms


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


def main():
    """main program"""

    prof = util.Profiler()
    prof.tick("total")

    # just in case
    random.seed(1)
    np.random.seed(1)

    width = 800
    height = 800

    output_dirname = "curl_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = output_dirname + ".mp4"
    output_poss_filename = "positions.pkl"

    fps = 30
    n_frames = 800  # 400
    n_particles = 800  # 1000
    trail_length = 50.0
    field_zoom = 1.0
    step = 0.05
    field_translation_rate = 0.01  # slight upward pull
    ramp_scale = 15.0  # 10-40, most testing was with 20
    frame_skip = 2     # lazy
    particle_size = 0  # 2
    glow_strength = 5.0  # 1.0 for particles of size 2, 5.0 for 0
    max_life = 180   # 180
    depth_sort = False

    calculate_positions = False

    cam_pos = np.array([0.0, 8.0, 20.0])
    view_pos = np.array([0.0, 0.0, 800.0])

    p_shift = np.array([width * 0.5, height * 0.5])[:, np.newaxis]

    os.makedirs(output_dirname, exist_ok=True)

    def ramp(x):
        """smooth ramp function from curl noise paper"""
        if x > 1.0:
            return 1.0
        elif x < -1.0:
            return -1.0
        else:
            return 15.0 / 8.0 * x - 10.0 / 8.0 * x * x * x + 3.0 / 8.0 * x * x * x * x * x

    nx = OpenSimplex(seed=1)
    ny = OpenSimplex(seed=2)
    nz = OpenSimplex(seed=3)

    # experiment with how to generate different velocities
    # i.e, constant y velocity is x -> z and z -> -x (or flip signs)

    # cx = lambda x, y, z: z
    # cy = lambda x, y, z: 0.0
    # cz = lambda x, y, z: -x

    def build_potential_funcs(frame_idx):
        sh = frame_idx * field_translation_rate

        def cx(x, y, z):
            noise = nx.noise3d(x, y + sh, z)
            struct = z
            a = ramp(math.sqrt(x * x + z * z) / ramp_scale)
            return a * noise + (1.0 - a) * struct

        def cy(x, y, z):
            noise = ny.noise3d(x, y + sh, z)
            struct = 0.0
            a = ramp(math.sqrt(x * x + z * z) / ramp_scale)
            return a * noise + (1.0 - a) * struct

        def cz(x, y, z):
            noise = nz.noise3d(x, y + sh, z)
            struct = -x
            a = ramp(math.sqrt(x * x + z * z) / ramp_scale)
            return a * noise + (1.0 - a) * struct

        return cx, cy, cz

    if calculate_positions:

        # I think I only need to save positions and ages, not velocities

        p_pos = (np.random.rand(n_particles, 3) - 0.5)  # * 10.0
        p_pos = p_pos / np.linalg.norm(p_pos, axis=1, keepdims=True)
        p_vel = np.zeros((n_particles, 3))
        p_active = np.ones(n_particles, dtype=np.int) * -1

        p_poss = [(p_pos, p_active)]

        for frame_idx in range(n_frames):
            print(frame_idx + 1, "/", n_frames, "\t", end="")

            # update particle positions
            p_pos = np.copy(p_pos)
            p_active = np.copy(p_active)

            # apply a rule for making particles active or inactive

            if frame_idx < 400:
                # activate particles
                p_active[frame_idx * 2] = 0
                p_active[frame_idx * 2 + 1] = 0

            # deactivate particles of a certain age
            p_active[p_active > (max_life + trail_length)] = -1

            # display active particle count
            print(np.sum(p_active > -1))

            prof.tick("pos")

            # construct our potential functions (allows time-varying fields)
            cx, cy, cz = build_potential_funcs(frame_idx)

            for p_idx in range(n_particles):
                if p_active[p_idx] > -1:
                    pos = p_pos[p_idx, :]
                    pos_z = pos * field_zoom
                    curl = curl_field_vector(pos_z[0], pos_z[1], pos_z[2], cx, cy, cz)
                    # print(curl)
                    p_vel[p_idx, :] = curl
                    p_pos[p_idx, :] = pos + p_vel[p_idx, :] * step
                    p_active[p_idx] += 1

            p_poss.append((p_pos, p_active))

            prof.tock("pos")

        with open(output_poss_filename, "wb") as output_poss_file:
            pickle.dump(p_poss, output_poss_file)

    else:
        with open(output_poss_filename, "rb") as input_poss_file:
            p_poss = pickle.load(input_poss_file)

    # render particles

    actives = np.concatenate([x[1] for x in p_poss], axis=0)

    for frame_idx in range(len(p_poss)):
        print(frame_idx + 1, "/", n_frames)

        prof.tick("trans")

        # TODO: only keep the last X for the trail length of the oldest particle?

        # calculate camera transform

        # static angle
        # rot = np.identity(3)
        # cam_trans = transforms.camera_transform(rot, cam_pos)

        angle = (frame_idx * 0.25 * np.pi) / n_frames
        cam_trans = np.dot(
            transforms.transformation(np.identity(3), -cam_pos),
            util.rotation_y(angle))

        # list of arrays
        p_pos_ps = []

        for (p_pos, p_active) in p_poss[:(frame_idx + 1)]:  # render all positions up to and including  current position

            # apply camera and perspective transformations
            p_pos_c = transforms.transform(cam_trans, np.transpose(p_pos))
            p_pos_p = transforms.perspective(p_pos_c, view_pos)
            # align and convert to int
            # TODO: flip y properly
            p_pos_p = np.clip(np.array(p_pos_p + p_shift), -width * 2, height * 2)
            p_pos_p = np.array(p_pos_p, dtype=np.int)
            p_pos_p = np.transpose(p_pos_p)

            # TODO: build a list of particles to render, with depths
            p_pos_ps.append((p_pos_p, p_pos_c[2, :]))  # smaller z is farther away

        prof.tock("trans")

        # ~~~~ ~~~~ ~~~~ ~~~~

        prof.tick("draw")

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas_im = Image.fromarray(canvas)  # I think I can do this once
        canvas_draw = ImageDraw.Draw(canvas_im)

        # stack arrays

        coords = np.concatenate([x[0] for x in p_pos_ps], axis=0)
        depths = np.concatenate([x[1] for x in p_pos_ps], axis=0)
        actives_head = np.concatenate([p_poss[frame_idx][1] for _ in range(len(p_poss))], axis=0)

        if depth_sort:
            sorted_idxs = np.argsort(depths)
        else:
            sorted_idxs = range(coords.shape[0])

        for p_idx in sorted_idxs:

            p_age_tail = actives[p_idx]         # age at current tail position?
            p_age_head = actives_head[p_idx]    # age at head in current frame?

            if -1 < p_age_tail <= max_life and p_age_head > -1:

                age_diff = p_age_head - p_age_tail  # age difference between head and tail

                color_scale = max(trail_length - age_diff, 0) / trail_length
                color_scale = max(min(color_scale, 1.0), 0.0)
                color = (
                    int(255 * color_scale),
                    int(255 * color_scale * color_scale),
                    0)

                x, y = coords[p_idx, :]

                # cv2.circle(canvas, (x, y), particle_size, color, -1)

                if particle_size == 0:
                    canvas_draw.point([(x, y)], color)
                else:
                    canvas_draw.ellipse(
                        [x - particle_size, y - particle_size, x + particle_size, y + particle_size],
                        color)

        canvas = np.array(canvas_im)

        prof.tock("draw")

        prof.tick("eff")
        canvas = effects.glow(canvas, 63, glow_strength)
        prof.tock("eff")

        prof.tick("disk")
        if frame_idx % frame_skip == 0:
            cv2.imwrite(
                os.path.join(output_dirname, str(int(frame_idx / frame_skip)).rjust(5, "0") + ".png"),
                canvas)
        prof.tock("disk")

    command = util.ffmpeg_command(output_dirname, output_filename, width, height, fps)
    os.system(command)

    prof.tock("total")

    times = [(x, y) for x, (_, y) in prof.times.items()]

    for key, dur in times:
        print(key, "\t", round(dur, 3))


if __name__ == "__main__":
    main()
