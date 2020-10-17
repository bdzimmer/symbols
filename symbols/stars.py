"""

2D star geometry.

"""

# Copyright (c) 2020 Ben Zimmer. All rights reserved.

from typing import List, Tuple

from symbols import transforms


Point2D = Tuple[float, float]


def star_points(n: int, m: int) -> List[Point2D]:
    """find the points of a regular star polygon of radius 1.0 centered at (0.0, 0.0)"""

    # https://en.wikipedia.org/wiki/Regular_polygon#Regular_star_polygons

    circle_points = transforms.points_around_circle(n, 0.0, 1.0)
    point_idxs = [(x * m) % n for x in list(range(n))]

    return [circle_points[idx] for idx in point_idxs]


def line_loop(points: List[Point2D]) -> List[Tuple[Point2D, Point2D]]:
    """convert a list of points into a list point pairs forming a closed loop"""
    return list(zip(points, points[1:] + [points[0]]))
