"""

Test animation functions.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from functools import partial as P
import math
from datetime import datetime
import os
import random

import attr
import cv2
import numpy as np

from symbols import effects, util, animate2d, stars
from symbols import draw_cv, draw_cairo
from symbols import symbols
from symbols import func

SCRATCH_DIRNAME = os.path.join("test_scratch", "animate")


def test_integration():
    """test everything with some animating shapes."""

    os.makedirs(SCRATCH_DIRNAME, exist_ok=True)

    prof = util.Profiler()
    prof.tick("total")

    random.seed(1)

    height = 800
    width = 800
    fps = 30
    save = True
    time_increment = 0.25
    # do_filter = True
    # do_effect = True
    render_func = draw_cairo.render

    # ~~~~ derived config

    output_dirname = SCRATCH_DIRNAME
    output_filename = output_dirname + ".mp4"

    if not save:
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("output", (width // 2, height // 2))

    star_color = (255, 255, 0, 200)
    star_center = width // 2, height // 2
    star_radius = 256
    dur = 120

    points_0 = stars.star_points(7, 3)
    polyline_0 = symbols.Polyline(
        center=(0, 0),
        lines=[symbols.Line(x, y, None, None, None) for x, y in stars.line_loop(points_0)],
        joint_type="",
        closed=True,
        thickness=8,
        depth=0,
        color=star_color)
    star_animation_0 = symbols.AnimDurationMulti(
        polyline_0,
        "star_73",
        dur,
        mods=["pen", ("rotate", -math.pi * 0.5, math.pi * 0.5)])

    points_1 = stars.star_points(7, 2)
    polyline_1 = symbols.Polyline(
        center=(0, 0),
        lines=[symbols.Line(x, y, None, None, None) for x, y in stars.line_loop(points_1)],
        joint_type="",
        closed=True,
        thickness=24,
        depth=0,
        color=star_color)
    star_animation_1 = symbols.AnimDurationMulti(
        polyline_1,
        "star_72",
        dur,
        mods=["pen", ("rotate", 1.5 * math.pi, math.pi * 0.5)])

    animation = (
        star_animation_0,
        star_animation_1)

    # def effect_func(
    #         prim: symbols.Primitive,
    #         prim_animated: symbols.Primitive,
    #         label: str,
    #         time: float):
    #     """translate the animation to the center of the canvas"""
    #     return [symbols.translate_primitive(prim_animated, star_center)]

    effect_func = None

    # scale and translate
    post_func = func.pipe(
        P(symbols.scale_center_primitive, fac=star_radius, center=(0, 0)),
        # P(symbols.rotate_center_primitive, rad=frac * max_angle, center=(0, 0)),
        P(symbols.translate_primitive, trans=star_center))

    # ~~~~ render the animation ~~~~

    # 'animation' is basically a DAG of Animations
    # animation_duration: DAG of TimedAnim, where time is total duration
    # anim_starts:        DAG of TimedAnim, where time is start time
    # animation_flat:     flat list of TimedAnim with start time of each animation

    animation_duration = symbols.find_duration(animation)
    animation_starts = symbols.find_starts(animation_duration, 0)
    animation_flat = sorted(symbols.flatten(animation_starts), key=lambda x: x.time)

    # print(animation_starts)
    # for x in animation_flat:
    #     print(x)
    # exit()

    # find discrete timesteps in sequence
    # times = sorted(list(set([x.time for x in animation_flat])))

    end_time = (
        animation_flat[-1].time +
        animation_flat[-1].anim.duration +
        dur / 2)

    for idx, time in enumerate(np.arange(0, end_time, time_increment)):
        print(time, "/", end_time)

        # black opaque canvas
        canvas = np.zeros((height, width, 4), dtype=np.uint8)
        canvas[:, :, 3] = 255

        prof.tick("draw")
        to_render = animate2d.animate_frame(animation_flat, time, effect_func)
        to_render = [post_func(x) for x in to_render]
        render_func(canvas, to_render)
        prof.tock("draw")

        # if do_filter:
        #     prof.tick("filter")
        #     canvas = image_filter(canvas)
        #     prof.tock("filter")

        prof.tick("disk")
        cv2.imwrite(
            os.path.join(output_dirname, str(idx).rjust(5, "0") + ".png"),
            canvas[:, :, [2, 1, 0]])
        prof.tock("disk")

    prof.tick("ffmpeg")
    command = util.ffmpeg_command(
        output_dirname, output_filename, width, height, fps)
    os.system(command)
    prof.tock("ffmpeg")

    prof.tock("total")
    prof.summary()
