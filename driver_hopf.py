"""

Hopf Fibration images and animations.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from datetime import datetime
import os

import numpy as np
import cv2

from symbols import util, transforms


def main():
    """main program"""

    width = 800
    height = 800

    output_dirname = "hopf_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = output_dirname + ".mp4"

    fps = 30

    p_shift = np.array([width * 0.5, height * 0.5])[:, np.newaxis]
    # look down at origin from z = 5
    cam_trans = transforms.camera_transform(np.identity(3), np.array([0.0, 0.0, 4.0]))
    view_pos = np.array([0.0, 0.0, 200.0])

    # ~~~ test with a circle

    n_points = 128

    # 3D points
    points = transforms.points_around_circle(n_points, 0.0, 1.0)
    points = np.array(points)
    points = np.column_stack((points, np.zeros(len(points))))
    points = np.transpose(points)

    fiber_points = points

    # render the animation

    os.makedirs(output_dirname, exist_ok=True)

    for frame_idx, angle in enumerate(np.linspace(0.0, 2.0 * np.pi, 360)):
        print(frame_idx, angle)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        rot = np.dot(
            # util.rotate_x(angle)[0:3, 0:3],
            util.rotation_y(0.5)[0:3, 0:3],
            util.rotation_z(angle)[0:3, 0:3],
        )

        base_points = np.dot(rot, points)[:, ::2]

        pts_0 = []
        pts_1 = []
        depths = []
        colors = []

        for base_point_idx in range(base_points.shape[1]):

            base_point = base_points[:, base_point_idx]

            a, b, c = base_point

            cos_theta = fiber_points[0, :]
            sin_theta = fiber_points[1, :]

            fiber = 1.0 / np.sqrt(2.0 * (1.0 + c)) * np.row_stack([
                (1.0 + c) * cos_theta,
                a * sin_theta - b * cos_theta,
                a * cos_theta + b * sin_theta,
                (1.0 + c) * sin_theta
            ])

            w = fiber[0, :]
            fiber_proj = np.row_stack([
                fiber[1, :] / (1.0 - w),
                fiber[2, :] / (1.0 - w),
                fiber[3, :] / (1.0 - w),
            ])

            # animation transformation
            fiber_proj = transforms.transform(util.rotate_x(angle), fiber_proj)

            # caluclate colors via normals
            fiber_mean = np.mean(fiber_proj, axis=1, keepdims=True)
            diffs = fiber_proj - fiber_mean
            diffs_norm = diffs / np.linalg.norm(diffs, axis=0, keepdims=True)
            vec = np.array([0.0, 0.0, 1.0])[:, np.newaxis]
            diffs_norm_dot = np.sum(diffs_norm * vec, axis=0, keepdims=True)
            diffs_norm_dot = np.clip(diffs_norm_dot, 0.0, 1.0)
            colors_fiber = np.array([230.0, 230.0, 0.0])[:, np.newaxis] * diffs_norm_dot
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

            if True:  # color != (0.0, 0.0, 0.0):  # a proxy for backface
                cv2.line(
                    canvas,
                    pt_0,
                    pt_1,
                    color,
                    1,
                    cv2.LINE_AA)

        # base_points_p = perspective(base_points, cam_trans, view_pos)
        # base_points_p = np.array(base_points_p + p_shift, dtype=int)
        # draw_polyline(canvas, base_points_p)

        from driver_symbols import image_filter
        canvas = image_filter(canvas)

        if False:
            cv2.imshow("output", canvas)
            # cv2.waitKey(-1)
            cv2.waitKey(33)
        else:
            cv2.imwrite(os.path.join(output_dirname, str(frame_idx).rjust(5, "0") + ".png"), canvas)

    # TODO: replace with central definition of ffmpeg command in util
    command = (
        "ffmpeg -y -r " + str(fps) +
        " -f image2 -s " + str(width) + "x" + str(height) +
        " -i " + output_dirname + "/%05d.png " +
        "-threads 2 -vcodec libx264 -crf 25 -pix_fmt yuv420p " + output_filename)
    os.system(command)


if __name__ == "__main__":
    main()
