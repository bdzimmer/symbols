"""

2D geometry rendering.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from typing import List, Callable

import numpy as np

from symbols import symbols, draw


def draw_frame(
        canvas: np.ndarray,
        animation_flat: List[symbols.TimedAnim],
        time: float,
        effect_func: Callable,
        ) -> None:

    """draw a frame of an animation"""

    # filter TimedAnims by time
    anims = [x for x in animation_flat if time >= x.time]

    to_render = []

    for x in anims:
        print("\tprocess:", x.anim.label, "-", x.anim.primitive.__class__.__name__)

        anim: symbols.Animation = x.anim
        prim: symbols.Primitive = anim.primitive

        # do the base animation, creating a derived primitive

        if isinstance(anim, symbols.AnimDuration):
            if isinstance(prim, symbols.Line):
                # assuming AnimDuration for now
                frac = np.clip(float(time - x.time) / x.anim.head_duration, 0.0, 1.0)
                prim_animated = symbols.line_frac(prim, frac)
            elif isinstance(prim, symbols.Circle):
                # assuming AnimDuration for now
                frac = np.clip(float(time - x.time) / x.anim.head_duration, 0.0, 1.0)
                prim_animated = symbols.circle_frac(prim, frac)
                # print(
                #     prim_animated.start_angle * 360.0 / symbols.TAU,
                #     prim_animated.end_angle * 360.0 / symbols.TAU)
            else:
                print("No implementation for primitive type", anim.__class__)
                prim_animated = None
        else:
            print("No implementation for animation type", anim.__class__)
            prim_animated = None

        if prim_animated is not None:
            to_render.append(prim_animated)
            prims_addl = effect_func(prim, prim_animated, anim.label, time)
            to_render.extend(prims_addl)

    # TODO: sorting???

    to_render = sorted(to_render, key=lambda x: x.depth)

    for prim in to_render:
        print("\tdraw:", x.anim.primitive.__class__.__name__)

        if isinstance(prim, symbols.Line):
            draw.draw_line_cv(canvas, prim)
        elif isinstance(prim, symbols.Circle):
            draw.draw_circle_cv(canvas, prim)