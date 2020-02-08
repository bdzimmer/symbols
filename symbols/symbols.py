"""
S Y M B O L S
"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

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


def circle_point(angle, radius):
    """get a point on a circle"""
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    return (x, y)


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
