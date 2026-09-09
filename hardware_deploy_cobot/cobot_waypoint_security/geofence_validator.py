"""Map-frame containment checks for indoor cobot navigation."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Iterable, Optional, Sequence, Tuple


Point = Tuple[float, float]


@dataclass(frozen=True)
class MapBoundary:
    min_x_m: float
    min_y_m: float
    max_x_m: float
    max_y_m: float

    @classmethod
    def from_config(cls, bounds: Sequence[float]) -> "MapBoundary":
        if len(bounds) != 4:
            raise ValueError("map_bounds_m must contain [min_x, min_y, max_x, max_y]")
        boundary = cls(*(float(value) for value in bounds))
        if boundary.min_x_m >= boundary.max_x_m or boundary.min_y_m >= boundary.max_y_m:
            raise ValueError("map bounds must have positive area")
        return boundary

    def contains(self, point: Point) -> bool:
        x_m, y_m = point
        return self.min_x_m <= x_m <= self.max_x_m and self.min_y_m <= y_m <= self.max_y_m


class RouteCorridor:
    """A polyline corridor supplied by the trusted mission planner."""

    def __init__(self, points: Iterable[Point], half_width_m: float) -> None:
        self.points = tuple((float(x), float(y)) for x, y in points)
        if len(self.points) < 2:
            raise ValueError("a route corridor needs at least two points")
        if half_width_m <= 0:
            raise ValueError("corridor half width must be positive")
        self.half_width_m = float(half_width_m)

    def contains(self, point: Point) -> bool:
        return self.distance_to(point) <= self.half_width_m

    def distance_to(self, point: Point) -> float:
        return min(
            _distance_to_segment(point, start, end)
            for start, end in zip(self.points, self.points[1:])
        )


def _distance_to_segment(point: Point, start: Point, end: Point) -> float:
    px, py = point
    ax, ay = start
    bx, by = end
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if denominator == 0:
        return hypot(px - ax, py - ay)
    ratio = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
    return hypot(px - (ax + ratio * dx), py - (ay + ratio * dy))
