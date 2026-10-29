import math
import random
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

from stretch_sim.geometry import Point, Pose2D, segments_intersect, world_to_local, wrap_angle
from stretch_sim.world import SimObject, WorldMap


@dataclass
class CameraModel:
    '''
    Which surface a scan covers: the nearest surface whose closest edge is within
    max_range of the robot and on its arm side (the side the robot parks facing), within
    side_margin of straight out to the right. The whole surface is scanned; where the
    head points is left to the perception stack, as on the robot.
    '''
    max_range: float = 1.5
    side_margin: float = math.radians(80.0)

    def scanned_surface(self, world: WorldMap, pose: Pose2D) -> Optional[str]:
        arm_side = wrap_angle(pose.yaw - math.pi / 2.0)
        best, best_d = None, math.inf
        for name, loc in world.locations.items():
            if loc.kind != "surface":
                continue
            r = loc.rect
            nx, ny = min(max(pose.x, r.x0), r.x1), min(max(pose.y, r.y0), r.y1)
            d = math.hypot(nx - pose.x, ny - pose.y)
            if d > self.max_range or d >= best_d:
                continue
            if d > 1e-6 and abs(wrap_angle(math.atan2(ny - pose.y, nx - pose.x) - arm_side)) > self.side_margin:
                continue
            if not line_of_sight(world, (pose.x, pose.y), (nx, ny)):
                continue
            best, best_d = name, d
        return best


@dataclass
class PerceptionNoise:
    false_label_prob: float = 0.0       # a seen object is reported under another object's label
    score_range: Tuple[float, float] = (0.6, 0.95)
    position_noise_m: float = 0.01

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> "PerceptionNoise":
        cfg = cfg or {}
        return cls(false_label_prob=float(cfg.get("false_label_prob", 0.0)),
                   score_range=tuple(cfg.get("score_range", (0.6, 0.95))),
                   position_noise_m=float(cfg.get("position_noise_m", 0.01)))


@dataclass
class Detection:
    '''
    One detected object, in the robot's base frame at the time of the scan (x forward,
    y left, z up), as the real detector reports it. true_object is ground truth for
    visualisation and tests only; it must never be passed to the robot's software.
    '''
    label: str
    score: float
    centroid: Tuple[float, float, float]
    size: Tuple[float, float, float]
    stamp: float
    grasp_direction: str = "front"      # "front" (grasp_carry) or "top" (topdown_grasp)
    true_object: str = field(default="", compare=False)
    world_xy: Point = field(default=(0.0, 0.0), compare=False)


@dataclass
class ScanResult:
    surface: Optional[str]              # the surface scanned, None if the robot was not at one
    detections: List[Detection]
    missed: List[str]                   # objects on the surface that were not reported (ground truth)


def line_of_sight(world: WorldMap, a: Point, b: Point) -> bool:
    return not any(segments_intersect(a, b, w0, w1) for w0, w1 in world.walls)


def detect(world: WorldMap, camera: CameraModel, pose: Pose2D, objects: Iterable[SimObject],
           noise: PerceptionNoise, rng: random.Random, stamp: float, miss_prob: float) -> ScanResult:
    '''Scans the whole surface the robot is at; each object on it is missed with miss_prob.'''
    objects = list(objects)
    surface = camera.scanned_surface(world, pose)
    on_surface = sorted((o for o in objects if surface is not None and o.location == surface), key=lambda o: o.name)
    other_labels = sorted({o.label for o in objects})

    detections, missed = [], []
    for obj in on_surface:
        if rng.random() < miss_prob:
            missed.append(obj.name)
            continue
        label = obj.label
        if noise.false_label_prob > 0.0 and rng.random() < noise.false_label_prob:
            label = rng.choice([l for l in other_labels if l != obj.label] or [obj.label])
        lx, ly = world_to_local(pose, obj.x + rng.gauss(0.0, noise.position_noise_m),
                                obj.y + rng.gauss(0.0, noise.position_noise_m))
        detections.append(Detection(
            label=label,
            score=round(rng.uniform(*noise.score_range), 3),
            centroid=(lx, ly, obj.z + rng.gauss(0.0, noise.position_noise_m)),
            size=(obj.size, obj.size, obj.size),
            stamp=stamp,
            grasp_direction=obj.grasp_direction,
            true_object=obj.name,
            world_xy=(obj.x, obj.y),
        ))
    return ScanResult(surface, detections, missed)
