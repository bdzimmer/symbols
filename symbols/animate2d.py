"""

2D geometry rendering.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from typing import List, Callable, Optional

import numpy as np

from symbols import symbols


def animate_frame(
        animation_flat: List[symbols.TimedAnim],
        time: float,
        effect_func: Optional[Callable]
        ) -> List[symbols.Primitive]:

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
            # basically just a single pen type
            frac = np.clip(
                float(time - timed_anim.time) / timed_anim.anim.head_duration, 0.0, 1.0)
            prim_animated = symbols.frac_primitive(prim, frac)

        elif isinstance(anim, symbols.AnimDurationMulti):
            frac = np.clip(
                float(time - timed_anim.time) / timed_anim.anim.duration, 0.0, 1.0)
            prim_animated = prim
            for mod in anim.mods:
                if mod == "pen":
                    prim_animated = symbols.frac_primitive(prim_animated, frac)
                elif mod[0] == "translate":
                    _, t_start, t_end = mod
                    t_cur = symbols.interp(t_start, t_end, frac)
                    prim_animated = symbols.translate_primitive(prim_animated, t_cur)
                elif mod[0] == "scale":
                    _, fac_start, fac_end = mod
                    fac_cur = fac_start + frac * (fac_end - fac_start)
                    if isinstance(prim, symbols.Line):
                        center_cur = (0.0, 0.0)
                    else:
                        center_cur = prim.center
                    prim_animated = symbols.scale_center_primitive(prim_animated, fac_cur, center_cur)
                elif mod[0] == "scale_center":
                    _, fac_start, fac_end, center_start, center_end = mod
                    fac_cur = fac_start + frac * (fac_end - fac_start)
                    center_cur = symbols.interp(center_start, center_end, frac)
                    prim_animated = symbols.scale_center_primitive(prim_animated, fac_cur, center_cur)
                elif mod[0] == "rotate":
                    _, rad_start, rad_end = mod
                    rad_cur = rad_start + frac * (rad_end - rad_start)
                    if isinstance(prim, symbols.Line):
                        center_cur = (0.0, 0.0)
                    else:
                        center_cur = prim.center
                    prim_animated = symbols.rotate_center_primitive(prim_animated, rad_cur, center_cur)
                elif mod[0] == "rotate_center":
                    _, rad_start, rad_end, center_start, center_end = mod
                    rad_cur = rad_start + frac * (rad_end - rad_start)
                    center_cur = symbols.interp(center_start, center_end, frac)
                    prim_animated = symbols.rotate_center_primitive(prim_animated, rad_cur, center_cur)
        else:
            print("No implementation for animation type", anim.__class__)
            prim_animated = None

        if prim_animated is not None:
            to_render.append(prim_animated)
            if effect_func is not None:
                prims_addl = effect_func(prim, prim_animated, anim.label, time)
                to_render.extend(prims_addl)

    to_render = sorted(to_render, key=lambda x: x.depth)

    return to_render
