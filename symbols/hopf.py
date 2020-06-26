"""

Hopf fibration functions.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import cv2
import numpy as np

from symbols import transforms


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
        lighting_func, line_width,     # visual appearance
        decoration_func,               # generate additional stuff on fibers
        decoration_only
        ):
    """for backward compatibility with older scripts"""

    pts, norms, idxs_0, idxs_1 = fibration_geom(
        base_points, fiber_points,
        decoration_func,
        decoration_only)

    obj_rot = transforms.sep_rotation(obj_trans)
    pts = transforms.transform(obj_trans, pts)
    norms = np.dot(obj_rot, norms)

    canvas_shape = (canvas.shape[1], canvas.shape[0])

    pts_0, pts_1, colors, depths = light_and_flatten_geometry(
        pts, norms, idxs_0, idxs_1, lighting_func,
        cam_trans, view_pos, canvas_shape)

    # sort by depths
    sorted_idxs = np.argsort(depths)

    draw_segments(
        canvas,
        pts_0[:, sorted_idxs],
        pts_1[:, sorted_idxs],
        colors[:, sorted_idxs],
        line_width)


def fibration_geom(
        base_points,
        fiber_points,
        decoration_func,               # generate additional stuff on fibers
        decoration_only
        ):
    """draw the Hopf fibration for a set of base points and fiber points"""

    fiber_proj_all = []
    fiber_proj_norms_all = []
    point_idxs_0_all = []
    point_idxs_1_all = []
    points_count = 0

    for base_point_idx in range(base_points.shape[1]):
        base_point = base_points[:, base_point_idx]

        # fibration and stereographic projection
        fiber = total_points(base_point, fiber_points)
        fiber_proj = stereographic(fiber)

        # stack lines and depths
        point_idxs = list(range(fiber_proj.shape[1])) + [0]
        point_idxs_0 = np.array(point_idxs[0:-1])
        point_idxs_1 = np.array(point_idxs[1:])

        # find normals
        fiber_mean = np.mean(fiber_proj, axis=1, keepdims=True)
        diffs = fiber_proj - fiber_mean
        fiber_proj_norms = diffs / np.linalg.norm(diffs, axis=0, keepdims=True)

        if decoration_func is not None:
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

            if decoration_only:
                start_idx = 0
            else:
                start_idx = fiber_proj.shape[1]

            # append a disk for each line segment

            point_idxs_0_addl = []
            point_idxs_1_addl = []
            points_addl = []
            norms_addl = []

            for idx in range(n_fiber_proj):
                disk_points, disk_idxs_0, disk_idxs_1, disk_seg_norms = decoration_func(
                    mean_to_centers[:, idx],
                    diffs[:, idx])

                disk_points = disk_points.T + centers[:, idx:(idx + 1)]
                disk_seg_norms = disk_seg_norms.T

                point_idxs_0_addl.append(disk_idxs_0 + start_idx)
                point_idxs_1_addl.append(disk_idxs_1 + start_idx)
                points_addl.append(disk_points)
                norms_addl.append(disk_seg_norms)

                start_idx = start_idx + disk_points.shape[1]

            # stack and concatenate
            point_idxs_0_addl = np.concatenate(point_idxs_0_addl, axis=0)
            point_idxs_1_addl = np.concatenate(point_idxs_1_addl, axis=0)
            points_addl = np.concatenate(points_addl, axis=1)
            norms_addl = np.concatenate(norms_addl, axis=1)

            if not decoration_only:
                point_idxs_0 = np.concatenate((point_idxs_0, point_idxs_0_addl), axis=0)
                point_idxs_1 = np.concatenate((point_idxs_1, point_idxs_1_addl), axis=0)
                fiber_proj = np.concatenate((fiber_proj, points_addl), axis=1)
                fiber_proj_norms = np.concatenate((fiber_proj_norms, norms_addl), axis=1)
            else:
                point_idxs_0 = point_idxs_0_addl
                point_idxs_1 = point_idxs_1_addl
                fiber_proj = points_addl
                fiber_proj_norms = norms_addl

        fiber_proj_all.append(fiber_proj)
        fiber_proj_norms_all.append(fiber_proj_norms)

        point_idxs_0_all.append(point_idxs_0 + points_count)
        point_idxs_1_all.append(point_idxs_1 + points_count)
        points_count = points_count + fiber_proj.shape[1]

    fiber_proj_all = np.concatenate(fiber_proj_all, axis=1)
    fiber_proj_norms_all = np.concatenate(fiber_proj_norms_all, axis=1)
    point_idxs_0_all = np.concatenate(point_idxs_0_all, axis=0)
    point_idxs_1_all = np.concatenate(point_idxs_1_all, axis=0)

    return fiber_proj_all, fiber_proj_norms_all, point_idxs_0_all, point_idxs_1_all


def light_and_flatten_geometry(
        pts, norms, idxs_0, idxs_1,
        lighting_func,
        cam_trans, view_pos, canvas_shape):

    """
    apply lighting calculation to 3D segments,
    apply perspective transformation,
    depth sort,
    and convert to 2D
    """

    width, height = canvas_shape
    p_shift = np.array([width * 0.5, height * 0.5])[:, np.newaxis]

    colors = lighting_func(pts, norms)

    # apply perspective transformation
    pts_c = transforms.transform(cam_trans, pts)
    pts_p = transforms.perspective(pts_c, view_pos)
    # align and convert to int
    # TODO: flip y properly
    pts_p[1, :] = 0.0 - pts_p[1, :]
    pts_p = np.array(pts_p + p_shift, dtype=np.int)

    pts_0 = pts_p[:, idxs_0]
    pts_1 = pts_p[:, idxs_1]

    # TODO: there might be a better way to do calculate depth, like max?
    # my gut says mean is better
    # depths = (pts[2, idxs_0] + pts[2, idxs_1]) * 0.5
    depths = (pts_c[2, idxs_0] + pts_c[2, idxs_1]) * 0.5

    return pts_0, pts_1, colors, depths


def lighting_func_generic(norms, color, power, transparent):
    """paramaterized sort of generic lighting function"""

    # compare normals against z-axis (direction camera is pointing)
    vec = np.array([0.0, 0.0, 1.0])[:, np.newaxis]

    # TODO: why am I not using dot here???
    norms_dot = np.sum(norms * vec, axis=0, keepdims=True)

    color_scale = np.clip(norms_dot, 0.0, 1.0)
    # color_scale = np.clip(np.abs(norms_dot), 0.0, 1.0)
    color_scale = color_scale ** power

    colors = np.array(color)[:, np.newaxis] * color_scale

    if transparent:
        colors[3, :] = color[3]
    else:
        colors[3, :] = 255.0

    return colors


def draw_segments(canvas, pts_0, pts_1, colors, line_width):
    """draw a collection of 2D segments on a canvas"""

    for line_idx in range(pts_0.shape[1]):

        color = tuple(colors[:, line_idx])
        pt_0 = tuple(pts_0[:, line_idx])
        pt_1 = tuple(pts_1[:, line_idx])

        print(pt_0, pt_1, color, line_width)

        # print(pt_0, "->", pt_1)
        # if True:  # color != (0.0, 0.0, 0.0):  # a proxy for backface culling
        cv2.line(
            canvas,
            pt_0,
            pt_1,
            color,
            line_width,
            cv2.LINE_AA)


def draw_segments_transparent(
        canvas, pts_0, pts_1, colors, line_width,
        composite,
        canvas_draw):

    if composite:
        canvas_comp = canvas
        # canvas_draw = np.zeros(canvas.shape, dtype=np.uint8)
    else:
        canvas_draw = canvas

    for line_idx in range(pts_0.shape[1]):

        color = tuple(colors[:, line_idx])
        pt_0 = tuple(pts_0[:, line_idx])
        pt_1 = tuple(pts_1[:, line_idx])

        cv2.line(
            canvas_draw,
            pt_0,
            pt_1,
            color,
            line_width,
            cv2.LINE_AA)

        if composite:
            x_0 = int(pt_0[0])
            y_0 = int(pt_0[1])
            x_1 = int(pt_1[0])
            y_1 = int(pt_1[1])

            x_l = min(x_0, x_1) - line_width - 1
            y_l = min(y_0, y_1) - line_width - 1
            x_r = max(x_0, x_1) + line_width + 1
            y_r = max(y_0, y_1) + line_width + 1

            src_chunk = canvas_draw[y_l:y_r, x_l:x_r, :]
            dst_chunk = canvas_comp[y_l:y_r, x_l:x_r, :]
            src_rgb = src_chunk[:, :, 0:3]
            dst_rgb = dst_chunk[:, :, 0:3]
            src_a = src_chunk[:, :, 3:4] / 255.0
            dst_a = dst_chunk[:, :, 3:4] / 255.0

            out_a = src_a + dst_a * (1.0 - src_a)
            out_rgb = src_rgb * src_a + dst_rgb * dst_a * (1.0 - src_a)
            out_rgb[out_a[:, :, 0] > 0.0] = (out_rgb / out_a)[out_a[:, :, 0] > 0.0]
            out_rgb[out_a[:, :, 0] == 0.0] = 0.0

            canvas_comp[y_l:y_r, x_l:x_r, 0:3] = np.array(out_rgb, dtype=np.uint8)
            canvas_comp[y_l:y_r, x_l:x_r, 3] = np.array(out_a[:, :, 0] * 255, dtype=np.uint8)

            # clear scratch
            canvas_draw[y_l:y_r, x_l:x_r, :] = 0.0


# TODO: refactor and fix
def draw_segment_transparent(
        pt_0, pt_1, color, line_width,
        composite,
        canvas_comp,  # canvas being drawn onto
        canvas_draw,   # temp canvas
        erase
    ):

    blur_size = 31

    if blur_size > 0:
        # set up for alpha blending
        color_mod = (color[0], color[1], color[2], 0)
        cv2.line(
            canvas_draw,
            pt_0,
            pt_1,
            color_mod,
            line_width + blur_size // 2,
            # cv2.LINE_AA
        )

    cv2.line(
        canvas_draw,
        pt_0,
        pt_1,
        color,
        line_width,
        # cv2.LINE_AA
    )

    if composite:
        x_0 = int(pt_0[0])
        y_0 = int(pt_0[1])
        x_1 = int(pt_1[0])
        y_1 = int(pt_1[1])

        if blur_size > 0:
            line_width = line_width + blur_size // 2

        x_l = min(x_0, x_1) - line_width - 1
        y_l = min(y_0, y_1) - line_width - 1
        x_r = max(x_0, x_1) + line_width + 1
        y_r = max(y_0, y_1) + line_width + 1

        src_chunk = canvas_draw[y_l:y_r, x_l:x_r, :]

        if blur_size > 0:
            blurred = cv2.GaussianBlur(
                src_chunk, (blur_size, blur_size), 0)
            if blurred is not None:
                # src_chunk = blurred
                src_chunk = np.clip(blurred * 2.0 + src_chunk, 0.0, 255.0)

        dst_chunk = canvas_comp[y_l:y_r, x_l:x_r, :]
        src_rgb = src_chunk[:, :, 0:3]
        dst_rgb = dst_chunk[:, :, 0:3]
        src_a = src_chunk[:, :, 3:4] / 255.0
        dst_a = dst_chunk[:, :, 3:4] / 255.0

        out_a = src_a + dst_a * (1.0 - src_a)
        out_rgb = src_rgb * src_a + dst_rgb * dst_a * (1.0 - src_a)
        out_rgb[out_a[:, :, 0] > 0.0] = (out_rgb / out_a)[out_a[:, :, 0] > 0.0]
        out_rgb[out_a[:, :, 0] == 0.0] = 0.0

        canvas_comp[y_l:y_r, x_l:x_r, 0:3] = np.array(out_rgb, dtype=np.uint8)
        canvas_comp[y_l:y_r, x_l:x_r, 3] = np.array(out_a[:, :, 0] * 255, dtype=np.uint8)

        # clear scratch
        if erase:
            canvas_draw[y_l:y_r, x_l:x_r, :] = 0.0


def disk_segments(x_axis, z_axis, radius, n_points):
    """disk segments"""
    # TODO: move this somewhere more generic, like a generic geometry module

    y_axis = np.cross(z_axis, x_axis)

    circle_points = np.array(transforms.points_around_circle(n_points, 0.0, radius))
    circle_points = np.concatenate([circle_points, np.zeros((n_points, 1))], axis=1)

    # build transformation matrix and transform

    t_mat = np.column_stack((x_axis, y_axis, z_axis))

    circle_points = np.dot(t_mat, circle_points.T).T

    range_n = list(range(n_points))

    idxs_0 = np.array(range_n)
    idxs_1 = np.array(range_n[1:] + [0])

    centers = 0.5 * (circle_points[idxs_1, :] + circle_points[idxs_0, :])
    normals = centers / np.linalg.norm(centers, axis=1, keepdims=True)

    return circle_points, idxs_0, idxs_1, normals
