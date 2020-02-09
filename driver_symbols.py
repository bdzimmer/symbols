"""

Driver / test program for symbols functionality.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from datetime import datetime
import os
import random

import cv2
import numpy as np

from symbols import effects, func, symbols
from symbols.symbols import TAU


def add(x, y):
    """add point tuples"""
    return x[0] + y[0], x[1] + y[1]


def to_int(x):
    """convert to integers"""
    return int(x[0]), int(x[1])


def line_frac(line, frac):
    """transform a line into a fraction of a line"""
    y_diff = line.end[1] - line.start[1]
    x_diff = line.end[0] - line.start[0]
    new_end = to_int(add(line.start, (frac * x_diff, frac * y_diff)))
    return symbols.Line(line.start, new_end, line.color, line.thickness)


def draw_line_cv(im, line):
    """draw a line on an image using OpenCV"""
    cv2.line(im, line.start, line.end, line.color, line.thickness, cv2.LINE_AA)


def draw_circle_cv(im, circle, frac):
    """draw a circle on an image using OpenCV"""

    # TODO: this seems to have some issues when frac = 0.0

    end_angle = frac * (circle.end_angle - circle.start_angle) * 360.0 / symbols.TAU

    cv2.ellipse(
        im, circle.center, (circle.radius, circle.radius),
        circle.start_angle * 360.0 / symbols.TAU,
        0.0,
        end_angle,
        circle.color,
        circle.thickness)


def points_around_circle(n_points, start, radius, center):
    return [
        func.apply(
            symbols.circle_point(x * TAU / n_points + start, radius),
            lambda p: add(p, center),
            to_int)
        for x in range(n_points)]


def image_filter(im):
    """CRT kind of effect"""

    im = effects.grid(im)
    glow_flicker = 1.0 + random.random()
    im = effects.glow(im, 63, glow_flicker)

    return im


def main():
    """main program"""

    random.seed(1)

    height = 800
    width = 800

    output_dirname = "render_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = output_dirname + ".mp4"

    fps = 30

    # cv2.namedWindow("output", cv2.WINDOW_NORMAL)

    # for now, we have a few different symbols we can test here

    symbol_name = ["seal", "crest"][0]

    if symbol_name == "seal":

        circle_points = points_around_circle(7, TAU / 4.0, 300.0, (400, 400))

        segment_idxs = [(x * 3) % 7 for x in list(range(7)) + [0]]
        segment_idxs = list(zip(segment_idxs[:-1], segment_idxs[1:]))

        lines_sequence = [
            symbols.Line(circle_points[x[0]], circle_points[x[1]], (0, 0, 200), 3)
            for x in segment_idxs]

        half_circle = symbols.Circle(
            to_int((width / 2, height / 2)),
            300,
            TAU * 0.75,
            TAU * 0.25,
            (0, 0, 200),
            3
        )

        half_circle_2 = symbols.Circle(
            to_int((width / 2, height / 2)),
            300,
            TAU * 1.25,
            TAU * 0.75,
            (0, 0, 200),
            3
        )

        dur = 12
        lines_animation = [symbols.AnimDuration(x, dur) for x in lines_sequence]
        lines_animation_timed = dur * len(lines_sequence)

        animation = (
            lines_animation,
            (symbols.AnimDuration(half_circle, lines_animation_timed),
             symbols.AnimDuration(half_circle_2, lines_animation_timed)))

    elif symbol_name == "crest":

        color = (200, 200, 0)
        center = (400, 300)

        triangle_points = points_around_circle(3, TAU * 0.25, 400.0, center)
        triangle_points_opp = points_around_circle(3, TAU * 0.25, -200.0, center)

        segment_idxs = [0, 1, 2, 0]
        segment_idxs = list(zip(segment_idxs[:-1], segment_idxs[1:]))

        border_sequence = [
            symbols.Line(triangle_points[x[0]], triangle_points[x[1]], color, 3)
            for x in segment_idxs]

        center_sequence = [
            symbols.Line(x, y, color, 3)
            for x, y in zip(triangle_points, triangle_points_opp)]

        circle = symbols.Circle(
            center,
            200,
            TAU * 0.25,
            TAU + TAU * 0.25,
            color,
            3
        )

        dur = 64

        animation = (
            tuple([symbols.AnimDuration(x, dur) for x in border_sequence]),
            tuple([symbols.AnimDuration(x, dur) for x in center_sequence]),
            symbols.AnimDuration(circle, dur))

    # ~~~~ render the animation ~~~~

    animation_duration = symbols.find_duration(animation)
    animation_starts = symbols.find_starts(animation_duration, 0)
    animation_flat = sorted(symbols.flatten(animation_starts), key=lambda x: x.time)

    # print(animation_starts)
    # for x in animation_flat:
    #     print(x)
    # exit()

    # find discrete timesteps in sequence
    # times = sorted(list(set([x.time for x in animation_flat])))

    end_time = animation_flat[-1].time + animation_flat[-1].anim.head_duration + 50

    os.makedirs(output_dirname, exist_ok=True)

    for idx, time in enumerate(np.arange(0, end_time, 0.25)):
        print(time)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        things_to_draw = [x for x in animation_flat if time >= x.time]
        for x in things_to_draw:
            prim = x.anim.primitive
            if isinstance(prim, symbols.Line):
                # assuming AnimDuration for now
                frac = np.clip(float(time - x.time) / x.anim.head_duration, 0.0, 1.0)
                lf = line_frac(prim, frac)
                draw_line_cv(canvas, lf)
            elif isinstance(prim, symbols.Circle):
                # assuming AnimDuration for now
                frac = np.clip(float(time - x.time) / x.anim.head_duration, 0.0, 1.0)
                draw_circle_cv(canvas, prim, frac)

        canvas = image_filter(canvas)

        if False:
            cv2.imshow("output", canvas)
            # cv2.waitKey(-1)
            cv2.waitKey(33)
        else:
            cv2.imwrite(os.path.join(output_dirname, str(idx).rjust(5, "0") + ".png"), canvas)

    # TODO: replace with command from util
    command = (
        "ffmpeg -y -r " + str(fps) +
        " -f image2 -s " + str(width) + "x" + str(height) +
        " -i " + output_dirname + "/%05d.png " +
        "-threads 2 -vcodec libx264 -crf 25 -pix_fmt yuv420p " + output_filename)
    os.system(command)


if __name__ == "__main__":
    main()