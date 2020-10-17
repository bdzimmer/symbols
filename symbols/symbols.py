"""
S Y M B O L S
"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from typing import Tuple, List, TypeVar
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


P = TypeVar("P", bound=Primitive, covariant=True)


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

    center = attr.ib()
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
    """An animated primitive"""
    primitive = attr.ib()
    label = attr.ib()


@attr.s(frozen=True)
class AnimDuration(Animation):
    """Primitive animated by drawing in sequence"""
    primitive = attr.ib()
    label = attr.ib()
    head_duration = attr.ib()


@attr.s(frozen=True)
class AnimDurationMulti(Animation):
    """Primitive with multple animations"""
    primitive = attr.ib()
    label = attr.ib()
    duration = attr.ib()
    mods = attr.ib()


@attr.s(frozen=True)
class TimedAnim:
    """An animation with a time (duration or start time)"""
    anim = attr.ib()
    time = attr.ib()


# ~~~~ utilities

def circle_point(angle, radius):
    """get a point on a circle"""
    x_coord = radius * np.cos(angle)
    y_coord = radius * np.sin(angle)
    return x_coord, y_coord


def add(pt_0, pt_1):
    """add point tuples"""
    return pt_0[0] + pt_1[0], pt_0[1] + pt_1[1]


def sub(pt_0, pt_1):
    """subract point tuples"""
    return pt_0[0] - pt_1[0], pt_0[1] - pt_1[1]


def scale(pnt, frac):
    """scale a point"""
    return pnt[0] * frac, pnt[1] * frac


def rotate(pnt, rad):
    """rotate a point"""
    return pnt[0] * math.cos(rad) - pnt[1] * math.sin(rad), pnt[0] * math.sin(rad) + pnt[1] * math.cos(rad)


def length(line: Line) -> float:
    """length of a line"""
    d_x = line.end[0] - line.start[0]
    d_y = line.end[1] - line.start[1]
    return math.sqrt(d_x * d_x + d_y * d_y)


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


def rotate_center(pnt, rad, center):
    """rotate a point around a center"""
    return add(rotate(sub(pnt, center), rad), center)


# ~~~~ animation utilities ~~~~


def radiate(time: float, num_facs: int, rate: float, maximum: float) -> List[float]:
    """generate cycling scale factors given time"""
    # used for animation

    range_max = maximum / rate
    range_step = range_max / num_facs

    return [
        1.0 + math.fmod((time + x) * rate, maximum)
        for x in np.arange(0.0, range_max, range_step)
    ]


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
    elif isinstance(struct, AnimDurationMulti):
        res = TimedAnim(struct, struct.duration)
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
        if isinstance(struct.anim, (tuple, list)):
            res = find_starts(struct.anim, start_time)
        else:
            res = TimedAnim(struct.anim, start_time)
        # TODO: for velocity, calculate time based on pen travel distance

    return res


def flatten(struct):
    """flatten a a nested structure of TimedAnims"""
    if isinstance(struct, tuple):
        res = [z for y in struct for z in flatten(y)]
    elif isinstance(struct, list):
        res = [z for y in struct for z in flatten(y)]
    elif isinstance(struct, TimedAnim):
        if isinstance(struct.anim, (tuple, list)):
            res = [flatten(struct.anim)]
        else:
            res = [struct]
    else:
        res = None

    return res


# ~~~~ dispatching functions for animating and transforming primitives ~~~~

def frac_primitive(prim: P, frac: float) -> P:
    """find a fraction of a primitive for animating the pen"""

    # TODO: consider dealing with the <=0.0 and >=1.0 cases here

    if isinstance(prim, Line):
        prim_animated = line_frac(prim, frac)
    elif isinstance(prim, Circle):
        prim_animated = circle_frac(prim, frac)
    elif isinstance(prim, Polyline):
        prim_animated = polyline_frac(prim, frac)
    else:
        print("No implementation of frac_primitive for primitive type", prim.__class__)
        prim_animated = None

    return prim_animated


def translate_primitive(prim: P, trans: Tuple) -> P:
    """translate"""

    if isinstance(prim, Line):
        prim_animated = attr.evolve(
            prim,
            start=add(prim.start, trans),
            end=add(prim.end, trans))
    elif isinstance(prim, Circle):
        prim_animated = attr.evolve(
            prim,
            center=add(prim.center, trans))
    elif isinstance(prim, Polyline):
        prim_animated = attr.evolve(
            prim,
            center=add(prim.center, trans),
            lines=[
                attr.evolve(
                    x,
                    start=add(x.start, trans),
                    end=add(x.end, trans))
                for x in prim.lines])
    else:
        print("No implementation of translate for primitive type", prim.__class__)
        prim_animated = None

    return prim_animated


def scale_center_primitive(prim: P, fac: float, center: Tuple) -> P:
    """scale a primitive around a center point"""

    if isinstance(prim, Line):
        prim_animated = attr.evolve(
            prim,
            start=scale_center(prim.start, fac, center),
            end=scale_center(prim.end, fac, center))
    elif isinstance(prim, Circle):
        prim_animated = attr.evolve(
            prim,
            radius=fac * prim.radius)
    elif isinstance(prim, Polyline):
        prim_animated = attr.evolve(
            prim,
            lines=[
                attr.evolve(
                    x,
                    start=scale_center(x.start, fac, center),
                    end=scale_center(x.end, fac, center))
                for x in prim.lines])
    else:
        print("No implementation of scale_center for primitive type", prim.__class__)
        prim_animated = None

    return prim_animated


def rotate_center_primitive(prim: P, rad: float, center: Tuple) -> P:
    """rotate a primitive"""

    if isinstance(prim, Line):
        prim_animated = attr.evolve(
            prim,
            start=rotate_center(prim.start, rad, center),
            end=rotate_center(prim.end, rad, center))
    elif isinstance(prim, Circle):
        prim_animated = attr.evolve(
            prim,
            center=rotate_center(prim.center, rad, center),
            start_angle=prim.start_angle + rad,
            end_angle=prim.end_angle + rad)
    elif isinstance(prim, Polyline):
        prim_animated = attr.evolve(
            prim,
            lines=[
                attr.evolve(
                    x,
                    start=rotate_center(x.start, rad, center),
                    end=rotate_center(x.end, rad, center))
                for x in prim.lines])
    else:
        print("No implementation of rotate_center for primitive type", prim.__class__)
        prim_animated = None

    return prim_animated
