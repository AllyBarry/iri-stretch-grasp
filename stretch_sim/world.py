import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from stretch_sim.config import default_priors_file, load_yaml
from stretch_sim.geometry import Point, Pose2D, Rect, yaw_with_target_on_right

# Where an object is while the robot holds it.
HELD = "Robot"
GRASP_DIRECTIONS = ("front", "top")   # the grasp_carry and topdown_grasp servers

_SIDES = {"north": (0.0, 1.0), "south": (0.0, -1.0), "east": (1.0, 0.0), "west": (-1.0, 0.0)}


@dataclass
class Location:
    '''
    A named place: a surface objects can stand on (Fridge, Table, ...) or a person (Human).
    approach is the base pose the robot drives to, with the location on its arm side.
    '''
    name: str
    kind: str                      # "surface" | "person"
    rect: Rect
    height: float                  # surface height; for a person, the hand height
    approach: Pose2D
    approach_side: Tuple[float, float]  # unit vector from the location towards the robot
    color: str = "#d8c9a8"
    slot_spacing: float = 0.12


@dataclass
class WorldMap:
    '''The static apartment.'''
    name: str
    width: float
    height: float
    walls: List[Tuple[Point, Point]]
    locations: Dict[str, Location]
    obstacles: List[Tuple[str, Rect]]   # furniture that is not a location, e.g. a sofa
    robot_start: Pose2D
    obstacle_colors: Dict[str, str] = field(default_factory=dict)

    def location_poses(self) -> Dict[str, Dict[str, float]]:
        '''The approach poses in the format of config/locations/location_poses.json.'''
        return {name: {k: round(v, 3) for k, v in loc.approach.to_dict().items()}
                for name, loc in self.locations.items()}

    def blocking_rects(self) -> List[Rect]:
        return [loc.rect for loc in self.locations.values()] + [rect for _, rect in self.obstacles]

    def location_at_pose(self, pose: Pose2D, tolerance: float = 0.15) -> Optional[str]:
        '''The location whose approach pose this is, if any (navigation goals arrive as poses).'''
        for name, loc in self.locations.items():
            if math.hypot(loc.approach.x - pose.x, loc.approach.y - pose.y) <= tolerance:
                return name
        return None

    def location_at(self, x: float, y: float, margin: float = 0.05) -> Optional[str]:
        for name, loc in self.locations.items():
            if loc.rect.contains(x, y, margin):
                return name
        return None


@dataclass
class SimObject:
    name: str
    category: str                  # e.g. "Ingredients" or "Tools"
    location: str                  # a Location name, or HELD
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0                 # height of the object's centre
    size: float = 0.07             # width across the gripper
    label: str = ""                # what the detector calls it
    grasp_direction: str = "front" # how the detector says to grasp it: "front" or "top"

    def copy(self) -> "SimObject":
        return SimObject(**self.__dict__)


def _approach_pose(rect: Rect, cfg: dict) -> Tuple[Pose2D, Tuple[float, float]]:
    side = cfg.get("side", "south")
    if side not in _SIDES:
        raise ValueError(f"Unknown approach side '{side}', expected one of {list(_SIDES)}")
    sx, sy = _SIDES[side]
    if "x" in cfg and "y" in cfg:
        x, y = float(cfg["x"]), float(cfg["y"])
        yaw = float(cfg["yaw"]) if "yaw" in cfg else yaw_with_target_on_right(rect.cx - x, rect.cy - y)
        return Pose2D(x, y, yaw), (sx, sy)

    standoff = float(cfg.get("standoff", 0.35))
    half = rect.d / 2.0 if sy else rect.w / 2.0
    x = rect.cx + sx * (half + standoff) + float(cfg.get("offset_x", 0.0))
    y = rect.cy + sy * (half + standoff) + float(cfg.get("offset_y", 0.0))
    return Pose2D(x, y, yaw_with_target_on_right(-sx, -sy)), (sx, sy)


def load_world(path: Path) -> WorldMap:
    cfg = load_yaml(path)

    walls = [((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))) for a, b in cfg.get("walls", [])]

    locations = {}
    for name, loc in cfg["locations"].items():
        rect = Rect(*[float(v) for v in loc["rect"]])
        approach, side = _approach_pose(rect, loc.get("approach", {}))
        locations[name] = Location(
            name=name,
            kind=loc.get("kind", "surface"),
            rect=rect,
            height=float(loc.get("height", 0.75)),
            approach=approach,
            approach_side=side,
            color=loc.get("color", "#d8c9a8" if loc.get("kind", "surface") == "surface" else "#f2c6a0"),
            slot_spacing=float(loc.get("slot_spacing", 0.12)),
        )

    obstacles = [(o.get("name", "obstacle"), Rect(*[float(v) for v in o["rect"]])) for o in cfg.get("obstacles") or []]
    obstacle_colors = {o.get("name", "obstacle"): o["color"] for o in cfg.get("obstacles") or [] if "color" in o}
    start = cfg.get("robot_start", {})

    return WorldMap(
        name=cfg.get("name", Path(path).stem),
        width=float(cfg["size"][0]),
        height=float(cfg["size"][1]),
        walls=walls,
        locations=locations,
        obstacles=obstacles,
        robot_start=Pose2D(float(start.get("x", 1.0)), float(start.get("y", 1.0)), float(start.get("yaw", 0.0))),
        obstacle_colors=obstacle_colors,
    )


'''
SPAWNING
'''
def load_priors(path: Optional[Path] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
    '''{"Ingredients": {"Basil": {"Fridge": 0.9, ...}, ...}, "Tools": {...}}'''
    with Path(path or default_priors_file()).open("r", encoding="utf-8") as f:
        return json.load(f)


def surface_slots(loc: Location, margin: float = 0.06) -> List[Point]:
    '''
    Places on a surface to stand objects, nearest to the robot's approach pose first, so
    the first objects to spawn are the ones in view and in reach of the arm.
    '''
    r, step = loc.rect, loc.slot_spacing
    nx = max(1, int((r.w - 2 * margin) / step) + 1)
    ny = max(1, int((r.d - 2 * margin) / step) + 1)
    xs = [r.cx - (nx - 1) * step / 2.0 + i * step for i in range(nx)]
    ys = [r.cy - (ny - 1) * step / 2.0 + j * step for j in range(ny)]
    slots = [(x, y) for x in xs for y in ys]
    ax, ay = loc.approach.x, loc.approach.y
    return sorted(slots, key=lambda p: (math.hypot(p[0] - ax, p[1] - ay), p))


def person_slot(loc: Location, index: int) -> Point:
    '''Objects handed to a person stack up in a column beside them.'''
    return (loc.rect.x1 + 0.12, loc.rect.y1 - 0.1 - 0.12 * index)


def sample_location(probs: Dict[str, float], rng: random.Random) -> str:
    names = sorted(probs)
    weights = [max(0.0, float(probs[n])) for n in names]
    if sum(weights) <= 0.0:
        raise ValueError(f"All spawn probabilities are zero: {probs}")
    return rng.choices(names, weights=weights, k=1)[0]


def spawn_objects(world: WorldMap, scenario: dict, rng: random.Random) -> Dict[str, SimObject]:
    '''
    Decides where every object really is. By default each object's location is drawn
    from the location priors (the same distribution the robot believes); spawn.place
    pins chosen objects, e.g. {"Feta": "Pantry"} to test a belief that is wrong.
    '''
    spawn = scenario.get("spawn", {})
    priors = load_priors(spawn.get("priors_file"))
    place = spawn.get("place", {}) or {}
    labels = scenario.get("detector_labels", {}) or {}
    sizes = scenario.get("object_sizes", {}) or {}
    directions = scenario.get("grasp_directions", {}) or {}
    bad = {n: d for n, d in directions.items() if d not in GRASP_DIRECTIONS}
    if bad:
        raise ValueError(f"grasp_directions must be one of {list(GRASP_DIRECTIONS)}: {bad}")
    include = spawn.get("objects", "all")

    categories = {obj: cat for cat, objs in priors.items() for obj in objs}
    names = sorted(categories) if include == "all" else list(include)
    unknown = [n for n in list(place) + names if n not in categories]
    if unknown:
        raise ValueError(f"Objects not in the priors file: {sorted(set(unknown))}")

    chosen = {}
    for name in names:
        loc = place.get(name) or sample_location(priors[categories[name]][name], rng)
        if loc not in world.locations or world.locations[loc].kind != "surface":
            raise ValueError(f"'{name}' would spawn at '{loc}', which is not a surface in world '{world.name}'")
        chosen[name] = loc

    objects = {}
    free = {name: surface_slots(loc) for name, loc in world.locations.items() if loc.kind == "surface"}
    for name in names:
        loc = world.locations[chosen[name]]
        if not free[loc.name]:
            raise ValueError(f"No free slots left on '{loc.name}' for '{name}'")
        x, y = free[loc.name].pop(0)
        size = float(sizes.get(name, 0.07))
        objects[name] = SimObject(name=name, category=categories[name], location=loc.name,
                                  x=x, y=y, z=loc.height + size / 2.0, size=size,
                                  label=labels.get(name, name.lower()),
                                  grasp_direction=directions.get(name, "front"))
    return objects


def free_slot(world: WorldMap, objects: Dict[str, SimObject], location: str) -> Point:
    occupied = [(o.x, o.y) for o in objects.values() if o.location == location]
    for x, y in surface_slots(world.locations[location]):
        if all(math.hypot(x - ox, y - oy) > 1e-6 for ox, oy in occupied):
            return (x, y)
    raise ValueError(f"No free slots left on '{location}'")
