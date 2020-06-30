"""

2D geometry rendering.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from typing import List, Callable, Optional

import numpy as np

from symbols import symbols


def draw_frame(
        canvas: np.ndarray,
        animation_flat: List[symbols.TimedAnim],
        time: float,
        effect_func: Optional[Callable],
        render_func: Optional[Callable],
        ) -> None:

    """draw a frame of an animation"""

    # filter TimedAnims by time
    # could do this outside, but I think it makes sense to do it here
    # for convenience
    anims = [x for x in animation_flat if time >= x.time]

    to_render = []

    for timed_anim in anims:

        print(
            "\tprocess:",
            timed_anim.anim.label, "-",
            timed_anim.anim.primitive.__class__.__name__)

        anim: symbols.Animation = timed_anim.anim
        prim: symbols.Primitive = anim.primitive

        # do the base animation, creating a derived primitive

        if isinstance(anim, symbols.AnimDuration):
            if isinstance(prim, symbols.Line):
                # assuming AnimDuration for now
                frac = np.clip(
                    float(time - timed_anim.time) / timed_anim.anim.head_duration, 0.0, 1.0)
                prim_animated = symbols.line_frac(prim, frac)
            elif isinstance(prim, symbols.Circle):
                # assuming AnimDuration for now
                frac = np.clip(
                    float(time - timed_anim.time) / timed_anim.anim.head_duration, 0.0, 1.0)
                prim_animated = symbols.circle_frac(prim, frac)
            else:
                print("No implementation for primitive type", anim.__class__)
                prim_animated = None
        else:
            print("No implementation for animation type", anim.__class__)
            prim_animated = None

        if prim_animated is not None:
            to_render.append(prim_animated)
            if effect_func is not None:
                prims_addl = effect_func(prim, prim_animated, anim.label, time)
                to_render.extend(prims_addl)

    to_render = sorted(to_render, key=lambda x: x.depth)

    if render_func is not None:
        render_func(canvas, to_render)
