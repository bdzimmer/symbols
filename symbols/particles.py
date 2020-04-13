"""

Particle emitters and related functionality.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

# ~~~~ particle emitters ~~~~

import random
from typing import List, Tuple, Callable

import numpy as np

from symbols import symbols


# type aliases
Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


def build_vertex_emitter(lines: List[symbols.Line]) -> Callable[[], Point2D]:
    """given a set of lines, choose one of the vertices"""
    def emit():
        """emit a particle"""
        line = random.choice(lines)
        if random.random() > 0.5:
            return line.start
        else:
            return line.end
    return emit


def build_line_emitter(lines: List[symbols.Line]) -> Callable[[], Point2D]:
    """given a set of lines, choose a point on one of the lines"""

    # build probabilties based on line length
    lengths = [symbols.length(x) for x in lines]
    # TODO: optional weights on probs!
    lengths_total = sum(lengths)
    probs = [x / lengths_total for x in lengths]

    idxs = list(range(len(lines)))

    def emit():
        """emit a particle"""
        # choose a line
        line_idx = np.random.choice(idxs, p=probs)
        line = lines[line_idx]
        # interpolate along line
        return symbols.interp(line.start, line.end, random.random())

    return emit
