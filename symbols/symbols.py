"""
S Y M B O L S
"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

import math

import attr
import numpy as np

TAU = 2.0 * np.pi


@attr.s(frozen=True)
class Primitive:
    """ABC for geometric primitives"""
    thickness = attr.ib()
    depth = attr.ib()
    color = attr.ib()


@attr.s(frozen=True)
class Line(Primitive):
    """a line"""
    start = attr.ib()
    end = attr.ib()

    thickness = attr.ib()
    depth = attr.ib()
    color = attr.ib()


@attr.s(frozen=True)
class Circle(Primitive):
    """a circle or arc"""
    center = attr.ib()
    radius = attr.ib()
    start_angle = attr.ib()
    end_angle = attr.ib()

    thickness = attr.ib()
    depth = attr.ib()
    color = attr.ib()


@attr.s(frozen=True)
class Polyline(Primitive):
    """a sequence of line segments"""
    lines = attr.ib()
    joint_type = attr.ib()
    closed = attr.ib()
    # TODO: cap type?

    thickness = attr.ib()
    depth = attr.ib()
    color = attr.ib()


@attr.s(frozen=True)
class Dot(Primitive):
    """a filled dot"""
    center = attr.ib()
    radius = attr.ib()

    thickness = attr.ib()
    depth = attr.ib()
    color = attr.ib()


# ~~~~ ~~~~ ~~~~ ~~~~

# if this isn't quite right, it's close.

@attr.s(frozen=True)
class Animation:
    primitive = attr.ib()
    label = attr.ib()


@attr.s(frozen=True)
class AnimVel(Animation):
    primitive = attr.ib()
    label = attr.ib()
    head_vel = attr.ib()  # head velocity
    tail_vel = attr.ib()  # tail velocity


@attr.s(frozen=True)
class AnimDuration(Animation):
    primitive = attr.ib()
    label = attr.ib()
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


def sub(x, y):
    """subract point tuples"""
    return x[0] - y[0], x[1] - y[1]


def scale(pnt, frac):
    """scale a point"""
    return pnt[0] * frac, pnt[1] * frac


def to_int(x):
    """convert to integers"""
    return int(x[0]), int(x[1])


def length(line: Line) -> float:
    """length of a line"""
    dx = line.end[0] - line.start[0]
    dy = line.end[1] - line.start[1]
    return math.sqrt(dx * dx + dy * dy)


def line_frac(line: Line, frac: float) -> Line:
    """transform a line into a fraction of a line"""
    new_end = interp(line.start, line.end, frac)
    return attr.evolve(line, end=new_end)


def circle_frac(circle: Circle, frac: float) -> Circle:
    """transform a circle into a fraction of a circle"""
    new_end = circle.start_angle + frac * (circle.end_angle - circle.start_angle)
    return attr.evolve(circle, end_angle=new_end)


def polyline_frac(polyline: Polyline, frac: float) -> Polyline:
    """transform a polyline into a fraction of a polyline"""

    # edge cases so we don't have to deal with out of bounds
    # on the np.where below

    if frac == 1.0:
        return polyline

    if frac == 0.0:
        return attr.evolve(polyline, lines=[], closed=False)

    # find the length of each line
    # TODO: do this on polyline creation, since it's invariant
    lengths = [length(x) for x in polyline.lines]
    lengths_total = sum(lengths)
    lengths_weighted = [x / lengths_total for x in lengths]
    lengths_weighted_cumsum = np.cumsum([0.0] + lengths_weighted)

    # find the last index where frac is greater than weighted length
    idx_final = np.where(lengths_weighted_cumsum < frac)[0][-1]

    # fraction of final line
    # TODO: test this on some shapes where the lengths aren't equal
    frac_final = (frac - lengths_weighted_cumsum[idx_final]) / lengths_weighted[idx_final]

    # derive final line
    line_final = line_frac(polyline.lines[idx_final], frac_final)

    # build new line list
    lines = polyline.lines[:idx_final] + [line_final]

    # TODO: additional logic for little bits of lines for proper caps!

    print(lengths_weighted_cumsum)
    print(frac, idx_final, len(lines))
    print("----")

    # derive new polyline
    return attr.evolve(
        polyline,
        lines=lines,
        closed=frac >= 1.0)


def interp(start, end, frac):
    """interpolation"""
    # diff = end - start
    # return start + (frac * diff)
    diff = (end[0] - start[0], end[1] - start[1])
    return (
        start[0] + frac * diff[0],
        start[1] + frac * diff[1])


def scale_center(pnt, fac, center):
    """scale point in relation to a center"""
    return add(scale(sub(pnt, center), fac), center)


# ~~~~ functions for constructing animations ~~~~


def find_duration(struct):
    """transform a structure of Animations into TimedAnims
    where time represents duration."""

    if isinstance(struct, tuple):  # parallel events
        timed = [find_duration(y) for y in struct]
        res = tuple(timed)
    elif isinstance(struct, list):  # sequential events
        timed = [find_duration(y) for y in struct]
        res = timed
    elif isinstance(struct, AnimDuration):
        res = TimedAnim(struct, struct.head_duration)
    else:
        # TODO: for velocity, calculate time based on pen travel distance
        res = None

    return res


def find_starts(struct, start_time):
    """transform a structure of TimedAnims where time respresents
    duration into a structure where time represents start.
    """

    if isinstance(struct, tuple):  # parallel events
        timed = [find_starts(y, start_time) for y in struct]
        res = tuple(timed)
    elif isinstance(struct, list):  # sequential events
        cur_time = start_time
        timed = []
        for sub_stuct in struct:
            starts = find_starts(sub_stuct, cur_time)
            timed.append(starts)
            print(cur_time, sub_stuct.time)
            cur_time = cur_time + sub_stuct.time
        res = timed
    else:
        if isinstance(struct.anim, tuple) or isinstance(struct.anim, list):
            res = find_starts(struct.anim, start_time)
        else:
            res = TimedAnim(struct.anim, start_time)
        # TODO: for velocity, calculate time based on pen travel distance

    return res


def flatten(struct):
    if isinstance(struct, tuple):
        res = [z for y in struct for z in flatten(y)]
    elif isinstance(struct, list):
        res = [z for y in struct for z in flatten(y)]
    elif isinstance(struct, TimedAnim):
        if isinstance(struct.anim, tuple) or isinstance(struct.anim, list):
            res = [flatten(struct.anim)]
        else:
            res = [struct]
    else:
        res = None

    return res