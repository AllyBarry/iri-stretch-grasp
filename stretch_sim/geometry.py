import math
from dataclasses import dataclass
from typing import List, Tuple

Point = Tuple[float, float]


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def copy(self) -> "Pose2D":
        return Pose2D(self.x, self.y, self.yaw)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "yaw": self.yaw}


def world_to_local(pose: Pose2D, x: float, y: float) -> Point:
    '''World point to the robot's base frame (x forward, y left).'''
    dx, dy = x - pose.x, y - pose.y
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return (c * dx + s * dy, -s * dx + c * dy)


def local_to_world(pose: Pose2D, lx: float, ly: float) -> Point:
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return (pose.x + c * lx - s * ly, pose.y + s * lx + c * ly)


@dataclass(frozen=True)
class Rect:
    '''Axis-aligned rectangle: centre, width along x, depth along y.'''
    cx: float
    cy: float
    w: float
    d: float

    @property
    def x0(self) -> float:
        return self.cx - self.w / 2.0

    @property
    def x1(self) -> float:
        return self.cx + self.w / 2.0

    @property
    def y0(self) -> float:
        return self.cy - self.d / 2.0

    @property
    def y1(self) -> float:
        return self.cy + self.d / 2.0

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return self.x0 - margin <= x <= self.x1 + margin and self.y0 - margin <= y <= self.y1 + margin

    def corners(self) -> List[Point]:
        return [(self.x0, self.y0), (self.x1, self.y0), (self.x1, self.y1), (self.x0, self.y1)]


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    d1, d2 = _cross(q1, q2, p1), _cross(q1, q2, p2)
    d3, d4 = _cross(p1, p2, q1), _cross(p1, p2, q2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def point_segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = b[0] - a[0], b[1] - a[1]
    length_sq = ax * ax + ay * ay
    t = 0.0 if length_sq == 0.0 else max(0.0, min(1.0, ((p[0] - a[0]) * ax + (p[1] - a[1]) * ay) / length_sq))
    return math.hypot(p[0] - (a[0] + t * ax), p[1] - (a[1] + t * ay))


def yaw_with_target_on_right(dx: float, dy: float) -> float:
    '''
    Base yaw that puts the direction (dx, dy) on the robot's right (-y), which is the
    side the Stretch arm extends to.
    '''
    return math.atan2(dx, -dy)
