"""
S Y M B O L S
"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import math

import attr
import numpy as np


TAU = 2.0 * np.pi


@attr.s(frozen=True)
class Line:
    """a line"""
    start = attr.ib()
    end = attr.ib()

    color = attr.ib()
    thickness = attr.ib()


@attr.s(frozen=True)
class Circle:
    """a circle or arc"""
    center = attr.ib()
    radius = attr.ib()
    start_angle = attr.ib()
    end_angle = attr.ib()

    color = attr.ib()
    thickness = attr.ib()


@attr.s(frozen=True)
class Dot:
    """a filled dot"""
    center = attr.ib()
    radius = attr.ib()


@attr.s(frozen=True)
class AnimVel:
    primitive = attr.ib()
    head_vel = attr.ib()  # head velocity
    tail_vel = attr.ib()  # tail velocity


@attr.s(frozen=True)
class AnimDuration:
    primitive = attr.ib()
    head_duration = attr.ib()


@attr.s(frozen=True)
class TimedAnim:
    anim = attr.ib()
    time = attr.ib()


# ~~~~ utilities

def circle_point(angle, radius):
    """get a point on a circle"""
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return x, y


def add(x, y):
    """add point tuples"""
    return x[0] + y[0], x[1] + y[1]


def to_int(x):
    """convert to integers"""
    return int(x[0]), int(x[1])


# def points_around_circle(n_points, start, radius, center):
#     return [
#         func.apply(
#             circle_point(x * TAU / n_points + start, radius),
#             lambda p: add(p, center),
#             to_int  # TODO: get rid of this eventually
#         )
#         for x in range(n_points)]


def length(line: Line) -> float:
    """length of a line"""
    dx = line.end[0] - line.start[0]
    dy = line.end[1] - line.start[1]
    return math.sqrt(dx * dx + dy * dy)


def line_frac(line, frac):
    """transform a line into a fraction of a line"""
    new_end = interp(line.start, line.end, frac)
    return Line(line.start, new_end, line.color, line.thickness)


def interp(start, end, frac):
    """interpolate along a line"""
    x_diff = end[0] - start[0]
    y_diff = end[1] - start[1]
    return add(start, (frac * x_diff, frac * y_diff))



# ~~~~ functions for constructing animations ~~~~

def find_duration(x):
    if isinstance(x, tuple):  # parallel events
        timed = [find_duration(y) for y in x]
        return tuple(timed)
    elif isinstance(x, list): # sequential events
        timed = [find_duration(y) for y in x]
        return timed
    elif isinstance(x, AnimDuration):
        return TimedAnim(x, x.head_duration)
    # TODO: for velocity, calculate time based on pen travel distance


def find_starts(x, start_time):
    if isinstance(x, tuple):  # parallel events
        timed = [find_starts(y, start_time) for y in x]
        return tuple(timed)
    elif isinstance(x, list):  # sequential events
        cur_time = start_time
        timed = []
        for y in x:
            yt = find_starts(y, cur_time)
            timed.append(yt)
            print(cur_time, y.time)
            cur_time = cur_time + y.time
        return timed
    else:
        if isinstance(x.anim, tuple) or isinstance(x.anim, list):
            return find_starts(x.anim, start_time)
        else:
            return TimedAnim(x.anim, start_time)

    # TODO: for velocity, calculate time based on pen travel distance


# TODO: this has some issues with returns
def flatten(x):
    if isinstance(x, tuple):
        return [z for y in x for z in flatten(y)]
    elif isinstance(x, list):
        return [z for y in x for z in flatten(y)]
    elif isinstance(x, TimedAnim):
        if isinstance(x.anim, tuple) or isinstance(x.anim, list):
            return [flatten(x.anim)]
        else:
            return [x]
