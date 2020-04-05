"""

Driver for curl noise experimentation.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.


from datetime import datetime
import math
import os
import pickle
import random
import time

import cv2
import numpy as np
from opensimplex import OpenSimplex

from symbols import effects, util, transforms, noise


def main():
    """main program"""

    prof = util.Profiler()
    prof.tick("total")

    # just in case
    random.seed(1)
    np.random.seed(1)

    width = 1280  # 800
    height = 720  # 800

    cam_pos = np.array([0.0, 8.0, 20.0])      # 0 8 20
    view_pos = np.array([0.0, 0.0, 600.0])    # 0 0 800

    fps = 30
    n_frames = 800  # 400
    n_particles = 800  # 1000
    trail_length = 50.0
    field_zoom = 1.0
    step = 0.05
    field_translation_rate = 0.01  # slight upward pull
    ramp_scale = 15.0  # 10-40, most testing was with 15
    frame_skip = 2     # lazy
    particle_size = 0  # 2
    glow_strength = 3.5  # 1.0 for particles of size 2, 5.0 for 0
    max_life = 180   # 180
    depth_sort = False
    render_delay = 0.25     # introduce delay to avoid burning up CPU

    calculate_positions = True
    single_frame = None

    output_poss_filename = "positions.pkl"
    output_prefix = "curl_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    if single_frame is None:
        output_dirname = output_prefix
        frame_prefix = ""
        output_filename = output_dirname + ".mp4"
    else:
        output_dirname = ""
        frame_prefix = output_prefix
        output_filename = ""

    build_potential_funcs = build_potential_funcs_constructor(
        field_translation_rate, ramp_scale)

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
                    curl = noise.curl_field_vector(pos_z[0], pos_z[1], pos_z[2], cx, cy, cz)
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

    def generate_frame(frame_idx):
        """generate a single frame"""

        # calculate camera transform

        # static angle
        # rot = np.identity(3)
        # cam_trans = transforms.camera_transform(rot, cam_pos)

        # dynamic angle
        angle = (frame_idx * 0.25 * np.pi) / n_frames
        cam_trans = np.dot(
            transforms.transformation(np.identity(3), -cam_pos),
            util.rotation_y(angle))

        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas = noise.draw_frame(
            canvas,
            p_poss[0:(frame_idx + 1)],
            actives,
            cam_trans,
            view_pos,
            max_life,
            trail_length,
            particle_size,
            depth_sort,
            prof)

        prof.tick("eff")
        canvas = effects.glow(canvas, 63, glow_strength)
        prof.tock("eff")

        prof.tick("disk")
        cv2.imwrite(
            os.path.join(
                output_dirname,
                frame_prefix + str(int(frame_idx / frame_skip)).rjust(5, "0") + ".png"),
            canvas)
        prof.tock("disk")

    if single_frame is not None:
        # generate a single name named after the frame number
        generate_frame(single_frame)

    else:
        # animate
        os.makedirs(output_dirname, exist_ok=True)
        for frame_idx in range(len(p_poss)):
            print(frame_idx + 1, "/", n_frames)
            if frame_idx % frame_skip == 0:
                generate_frame(frame_idx)
                if render_delay > 0.0:
                    time.sleep(render_delay)

        command = util.ffmpeg_command(output_dirname, output_filename, width, height, fps)
        os.system(command)

    prof.tock("total")

    prof.summary()


def build_potential_funcs_constructor(field_translation_rate, ramp_scale):
    """build noise function constructor for the flame"""

    nx = OpenSimplex(seed=1)
    ny = OpenSimplex(seed=2)
    nz = OpenSimplex(seed=3)

    # here, we want constant y velocity in the curl field
    # so x -> z and z -> -x (or flip signs):
    #   cx = lambda x, y, z: z
    #   cy = lambda x, y, z: 0.0
    #   cz = lambda x, y, z: -x

    def build_potential_funcs(frame_idx):
        sh = frame_idx * field_translation_rate

        def cx(x, y, z):
            nv = nx.noise3d(x, y + sh, z)
            struct = z
            a = noise.ramp(math.sqrt(x * x + z * z) / ramp_scale)
            return a * nv + (1.0 - a) * struct

        def cy(x, y, z):
            nv = ny.noise3d(x, y + sh, z)
            struct = 0.0
            a = noise.ramp(math.sqrt(x * x + z * z) / ramp_scale)
            return a * nv + (1.0 - a) * struct

        def cz(x, y, z):
            nv = nz.noise3d(x, y + sh, z)
            struct = -x
            a = noise.ramp(math.sqrt(x * x + z * z) / ramp_scale)
            return a * nv + (1.0 - a) * struct

        return cx, cy, cz

    return build_potential_funcs


if __name__ == "__main__":
    main()
