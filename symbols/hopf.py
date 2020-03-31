"""

Hopf fibration functions.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import cv2
import numpy as np

from symbols import transforms
from symbols import symbols


def total_points(base_point, fiber_points):
    """find points in the total space S3 from a point in S2 and points S1"""

    a, b, c = base_point

    cos_theta = fiber_points[0, :]
    sin_theta = fiber_points[1, :]

    res = 1.0 / np.sqrt(2.0 * (1.0 + c)) * np.row_stack([
        (1.0 + c) * cos_theta,
        a * sin_theta - b * cos_theta,
        a * cos_theta + b * sin_theta,
        (1.0 + c) * sin_theta
    ])

    return res


def stereographic(points_4d):
    """project 4D points to 3D via the stereographic projection"""
    w = points_4d[0, :]
    points_3d = np.row_stack([
        points_4d[1, :] / (1.0 - w),
        points_4d[2, :] / (1.0 - w),
        points_4d[3, :] / (1.0 - w),
    ])
    return points_3d


def draw_fibration(
        canvas,
        base_points, fiber_points,
        obj_trans,                     # transform of object
        cam_trans, view_pos,           # camera and viewer positions
        color, line_width              # visual appearance
        ):

    """draw the Hopf fibration for a set of base points and fiber points"""

    # TODO: args for color

    rings = True

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
        fiber = total_points(base_point, fiber_points)
        fiber_proj = stereographic(fiber)

        # stack lines and depths
        point_idxs = list(range(fiber_proj.shape[1])) + [0]
        point_idxs_0 = point_idxs[0:-1]
        point_idxs_1 = point_idxs[1:]

        if rings:
            n_fiber_proj = fiber_proj.shape[1]

            # find z-axis of disks
            diffs = fiber_proj[:, point_idxs_1] - fiber_proj[:, point_idxs_0]
            diffs = diffs / np.linalg.norm(diffs, axis=0)

            # find x-axis of disks
            centers = (fiber_proj[:, point_idxs_0] + fiber_proj[:, point_idxs_1]) * 0.5
            means = np.tile(np.mean(centers, axis=1)[:, np.newaxis], (1, n_fiber_proj))
            mean_to_centers = centers - means
            mean_to_centers = mean_to_centers / np.linalg.norm(mean_to_centers, axis=0, keepdims=True)

            # was 2.7 sec

            start_idx = fiber_proj.shape[1]

            # append a disk for each line segment

            point_idxs_0_addl = []
            point_idxs_1_addl = []
            points_addl = []

            for idx in range(n_fiber_proj):
                disk_points, disk_idxs_0, disk_idxs_1 = disk_segments(
                    mean_to_centers[:, idx], diffs[:, idx], 0.1, 16)

                disk_points = disk_points.T + centers[:, idx:(idx + 1)]

                point_idxs_0_addl.append(disk_idxs_0 + start_idx)
                point_idxs_1_addl.append(disk_idxs_1 + start_idx)
                points_addl.append(disk_points)

                start_idx = start_idx + disk_points.shape[1]

            # stack and concatenate
            point_idxs_0_addl = np.concatenate(point_idxs_0_addl, axis=0)
            point_idxs_1_addl = np.concatenate(point_idxs_1_addl, axis=0)
            points_addl = np.concatenate(points_addl, axis=1)

            point_idxs_0 = np.concatenate((point_idxs_0, point_idxs_0_addl), axis=0)
            point_idxs_1 = np.concatenate((point_idxs_1, point_idxs_1_addl), axis=0)
            fiber_proj = np.concatenate((fiber_proj, points_addl), axis=1)

            # # a quick test
            # if False:
            #     # this gives us two axes -> three axes for a coordinate system
            #     # for each segment!
            #     spikes = centers + mean_to_centers * 0.05
            #     n_spikes = spikes.shape[1]
            #     # fiber_proj = np.concatenate([fiber_proj, means, centers], axis=1)
            #     fiber_proj = np.concatenate([fiber_proj, centers, spikes], axis=1)
            #
            #     point_0_idxs = np.concatenate([
            #         point_0_idxs, range(n_fiber_proj, n_fiber_proj + n_spikes)], axis=0)
            #     point_1_idxs = np.concatenate([
            #         point_1_idxs, range(n_fiber_proj + n_spikes, n_fiber_proj + n_spikes * 2)], axis=0)

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

        pts_0.append(fiber_proj_p[:, point_idxs_0])
        pts_1.append(fiber_proj_p[:, point_idxs_1])

        # TODO: there might be a better way to do this, like max?
        # my gut says mean is better
        depths_fiber = (fiber_proj_c[2, point_idxs_0] + fiber_proj_c[2, point_idxs_1]) * 0.5
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


def disk_segments(x_axis, z_axis, radius, n_points):
    """disk segments"""
    y_axis = np.cross(z_axis, x_axis)

    circle_points = np.array(transforms.points_around_circle(n_points, 0.0, radius))
    circle_points = np.concatenate([circle_points, np.zeros((n_points, 1))], axis=1)

    # build transformation matrix and transform

    t_mat = np.column_stack((x_axis, y_axis, z_axis))

    circle_points = np.dot(t_mat, circle_points.T).T

    range_n = list(range(n_points))

    idxs_0 = np.array(range_n)
    idxs_1 = np.array(range_n[1:] + [0])

    return circle_points, idxs_0, idxs_1
