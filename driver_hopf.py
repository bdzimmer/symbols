"""

Hopf Fibration images and animations.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from datetime import datetime
import os

import numpy as np
import cv2

from symbols import effects, hopf, transforms, util


def main():
    """main program"""

    width = 800
    height = 800
    fps = 30
    color = (50, 0, 200)  # (230, 230, 0)
    line_width = 2        # 1
    glow_size = 63
    glow_strength = 2.0
    n_points = 128        # number of points in each ring

    animation = False

    output_prefix = "hopf_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # look down at origin from z = 5
    cam_trans = transforms.camera_transform(np.identity(3), np.array([0.0, 0.0, 4.0]))
    view_pos = np.array([0.0, 0.0, 200.0])

    # 3D points in a flat circle in the XY plane

    circle_points = transforms.points_around_circle(n_points, 0.0, 1.0)
    circle_points = np.array(circle_points)
    circle_points = np.column_stack((circle_points, np.zeros(len(circle_points))))
    circle_points = np.transpose(circle_points)

    fiber_points = circle_points

    if not animation:

        # for now, a hard-coded configuration

        output_prefix = "hopf_20200215"
        output_filename = output_prefix + ".png"

        base_points = circle_points[:, 20:24:2]

        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        draw_fibration(
            canvas,
            base_points, fiber_points,
            np.identity(4),
            cam_trans, view_pos,
            color, line_width)

        cv2.imwrite(output_filename, canvas)

    else:

        output_dirname = output_prefix
        output_filename = output_dirname + ".mp4"

        # render the animation

        os.makedirs(output_dirname, exist_ok=True)

        for frame_idx, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, 360)):
            print(frame_idx, angle)

            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            # rotate a circle to get a set of base points

            rot = np.dot(
                # util.rotate_x(angle)[0:3, 0:3],
                util.rotation_y(0.5)[0:3, 0:3],
                util.rotation_z(angle)[0:3, 0:3])
            base_points = np.dot(rot, circle_points)[:, ::2]

            # draw the frame

            obj_trans = util.rotation_x(angle)
            draw_fibration(
                canvas,
                base_points, fiber_points,
                obj_trans,
                cam_trans, view_pos,
                color, line_width)

            # apply a glow effect

            canvas = effects.glow(canvas, glow_size, glow_strength)

            if False:
                cv2.imshow("output", canvas)
                # cv2.waitKey(-1)
                cv2.waitKey(33)
            else:
                cv2.imwrite(os.path.join(output_dirname, str(frame_idx).rjust(5, "0") + ".png"), canvas)

        command = util.ffmpeg_command(output_dirname, output_filename, width, height, fps)
        os.system(command)


def draw_fibration(
        canvas,
        base_points, fiber_points,
        obj_trans,                     # transform of object
        cam_trans, view_pos,           # camera and viewer positions
        color, line_width              # visual appearance
        ):

    """draw the Hopf fibration for a set of base points and fiber points"""

    # TODO: args for color

    color = np.array(color, dtype=np.float)[:, np.newaxis]

    height, width, _ = canvas.shape
    p_shift = np.array([width * 0.5, height * 0.5])[:, np.newaxis]

    pts_0 = []
    pts_1 = []
    depths = []
    colors = []

    for base_point_idx in range(base_points.shape[1]):
        base_point = base_points[:, base_point_idx]

        # fibration and stereographic projection
        fiber = hopf.total_points(base_point, fiber_points)
        fiber_proj = hopf.stereographic(fiber)

        # animation transformation
        fiber_proj = transforms.transform(obj_trans, fiber_proj)

        # caluclate colors via normals
        fiber_mean = np.mean(fiber_proj, axis=1, keepdims=True)
        diffs = fiber_proj - fiber_mean
        diffs_norm = diffs / np.linalg.norm(diffs, axis=0, keepdims=True)

        # compare normals against z-axis (direction camera is pointing)
        vec = np.array([0.0, 0.0, 1.0])[:, np.newaxis]
        diffs_norm_dot = np.sum(diffs_norm * vec, axis=0, keepdims=True)

        # sort of backface culling effect (segments "facing" away invisible)
        # color_scale = np.clip(diffs_norm_dot, 0.0, 1.0)

        # clip may not be necessary here
        color_scale = np.abs(diffs_norm_dot)
        color_scale = np.clip(color_scale, 0.0, 1.0)

        colors_fiber = color * color_scale
        colors_fiber = np.array(colors_fiber, np.float)
        colors.append(colors_fiber)

        # apply perspective transformation
        fiber_proj_c = transforms.transform(cam_trans, fiber_proj)
        fiber_proj_p = transforms.perspective(fiber_proj_c, view_pos)
        # align and convert to int
        # TODO: flip y properly
        fiber_proj_p = np.array(fiber_proj_p + p_shift, dtype=np.int)

        # stack lines and depths
        point_idxs = list(range(fiber_proj_p.shape[1])) + [0]
        point_0_idxs = point_idxs[0:-1]
        point_1_idxs = point_idxs[1:]

        pts_0.append(fiber_proj_p[:, point_0_idxs])
        pts_1.append(fiber_proj_p[:, point_1_idxs])

        # TODO: there might be a better way to do this, like max?
        # my gut says mean is better
        depths_fiber = (fiber_proj_c[2, point_0_idxs] + fiber_proj_c[2, point_1_idxs]) * 0.5
        depths.append(depths_fiber)

        # draw_polyline(canvas, fiber_proj_p, colors_fiber)

    # stack and draw all of them at once

    pts_0 = np.concatenate(pts_0, axis=1)
    pts_1 = np.concatenate(pts_1, axis=1)
    colors = np.concatenate(colors, axis=1)
    depths = np.concatenate(depths, axis=0)

    # sort by depths
    sorted_idxs = np.argsort(depths)

    for line_idx in sorted_idxs:  # range(pts_0.shape[1]):

        color = tuple(colors[:, line_idx])
        pt_0 = tuple(pts_0[:, line_idx])
        pt_1 = tuple(pts_1[:, line_idx])

        # print(pt_0, "->", pt_1)

        if True:  # color != (0.0, 0.0, 0.0):  # a proxy for backface culling
            cv2.line(
                canvas,
                pt_0,
                pt_1,
                color,
                line_width,
                cv2.LINE_AA)


if __name__ == "__main__":
    main()
