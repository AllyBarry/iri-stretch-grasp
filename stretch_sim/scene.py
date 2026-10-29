'''
Turns a snapshot of the simulation into a list of plain shapes in world coordinates
(metres, y up). Renderers (Tk window, PNG snapshot) only know how to draw shapes, so
the picture is the same everywhere and a new renderer is a few dozen lines.
'''
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from stretch_sim.geometry import Point, local_to_world
from stretch_sim.perception import CameraModel
from stretch_sim.robot import RobotGeometry
from stretch_sim.simulator import SimSnapshot
from stretch_sim.world import HELD, WorldMap

CATEGORY_COLORS = {"Ingredients": "#5fa845", "Tools": "#3f7fcf"}
DEFAULT_OBJECT_COLOR = "#999999"


@dataclass
class Shape:
    kind: str                         # "poly" | "line" | "circle" | "text"
    points: List[Point]
    fill: Optional[str] = None
    outline: Optional[str] = None
    width: float = 1.0                # pixels
    dash: bool = False
    alpha: float = 1.0
    radius: float = 0.0               # metres, for circles
    text: str = ""
    size: int = 9                     # font size, for text
    anchor: str = "center"            # "center" | "n" | "s" | "w" | "e"
    bold: bool = False


def abbreviations(names) -> Dict[str, str]:
    '''Short unique labels: "Basil" -> "Bas", "Olive Oil" -> "OlO", "Cutting Board" -> "CuB".'''
    out = {}
    for n in sorted(names):
        words = n.split()
        abbr = words[0][:3] if len(words) == 1 else words[0][:2] + "".join(w[0] for w in words[1:])
        length = len(abbr)
        while abbr in out.values():
            length += 1
            abbr = n.replace(" ", "")[:length]
        out[n] = abbr
    return out


def _rotated_square(cx: float, cy: float, yaw: float, size: float) -> List[Point]:
    h = size / 2.0
    c, s = math.cos(yaw), math.sin(yaw)
    return [(cx + c * x - s * y, cy + s * x + c * y) for x, y in ((h, h), (-h, h), (-h, -h), (h, -h))]


def build_scene(world: WorldMap, snap: SimSnapshot, geom: RobotGeometry, camera: CameraModel,
                show_labels: bool = True, highlight: Optional[str] = None) -> List[Shape]:
    shapes: List[Shape] = []
    labels = abbreviations(snap.objects)

    # Floor
    shapes.append(Shape("poly", [(0, 0), (world.width, 0), (world.width, world.height), (0, world.height)],
                        fill="#f6f3ec", outline=None))

    # Furniture that is not a location
    for name, rect in world.obstacles:
        shapes.append(Shape("poly", rect.corners(), fill=world.obstacle_colors.get(name, "#ddd6ca"), outline="#8f8a80"))
        shapes.append(Shape("text", [(rect.cx, rect.cy)], text=name, size=9, bold=True, fill="#4f4a40"))

    # Locations, with the pose the robot parks at
    for loc in world.locations.values():
        r = loc.rect
        if loc.kind == "person":
            shapes.append(Shape("circle", [(r.cx, r.cy)], radius=min(r.w, r.d) / 2.0, fill=loc.color, outline="#b07850", width=2))
            shapes.append(Shape("text", [(r.cx, r.cy)], text="☺", size=18, fill="#7a4a2a"))
        else:
            shapes.append(Shape("poly", r.corners(), fill=loc.color, outline="#8a7b62", width=2))
        shapes.append(Shape("text", [(r.cx, r.y1 + 0.06)], text=loc.name.upper(), size=9, bold=True,
                            fill="#5b4f3c", anchor="s"))
        a = loc.approach
        tip = (a.x + 0.18 * math.cos(a.yaw), a.y + 0.18 * math.sin(a.yaw))
        shapes.append(Shape("poly", _rotated_square(a.x, a.y, a.yaw, geom.base_size), outline="#c9bfae", dash=True))
        shapes.append(Shape("line", [(a.x, a.y), tip], outline="#c9bfae", width=1))

    # Walls
    for a, b in world.walls:
        shapes.append(Shape("line", [a, b], outline="#3a3a3a", width=5))

    robot = snap.robot

    # The surface the last scan covered
    if snap.scan and snap.scan.get("surface"):
        shapes.append(Shape("poly", world.locations[snap.scan["surface"]].rect.corners(), fill="#ffd54a",
                            outline="#e0b000", alpha=0.25, width=3))

    # Planned path
    if len(snap.path) >= 2:
        shapes.append(Shape("line", snap.path, outline="#2f7de1", width=2, dash=True))
        shapes.append(Shape("circle", [snap.path[-1]], radius=0.04, fill="#2f7de1"))

    # Objects on surfaces and with people
    detected = {d.true_object for d in snap.detections}
    missed = set(snap.scan["missed"]) if snap.scan else set()
    for obj in snap.objects.values():
        if obj.location == HELD:
            continue
        _draw_object(shapes, obj, labels, show_labels, detected, highlight, missed)

    # Robot
    pose = robot.pose
    body = _rotated_square(pose.x, pose.y, pose.yaw, geom.base_size)
    shapes.append(Shape("poly", body, fill="#3d4b5c", outline="#1f2833", width=2))
    front = [local_to_world(pose, geom.base_size / 2.0, 0.1), local_to_world(pose, geom.base_size / 2.0, -0.1)]
    shapes.append(Shape("line", front, outline="#f0f0f0", width=3))
    gx, gy = robot.gripper_xy(geom)
    shapes.append(Shape("line", [(pose.x, pose.y), (gx, gy)], outline="#9aa7b4", width=6))
    prong = 0.04 + robot.aperture / 2.0
    p1 = local_to_world(pose, prong, -(geom.arm_min_reach + robot.arm_extension) - 0.05)
    p2 = local_to_world(pose, -prong, -(geom.arm_min_reach + robot.arm_extension) - 0.05)
    shapes.append(Shape("line", [p1, (gx, gy), p2], outline="#1f2833", width=3))
    head = local_to_world(pose, 0.0, 0.0)
    look = (head[0] + 0.25 * math.cos(pose.yaw + robot.head_pan), head[1] + 0.25 * math.sin(pose.yaw + robot.head_pan))
    shapes.append(Shape("line", [head, look], outline="#e0b000", width=3))
    shapes.append(Shape("circle", [head], radius=0.05, fill="#e0b000", outline="#1f2833"))

    if robot.held:
        _draw_object(shapes, snap.objects[robot.held], labels, show_labels, detected, highlight, missed)

    return shapes


def _draw_object(shapes, obj, labels, show_labels, detected, highlight, missed=frozenset()):
    color = CATEGORY_COLORS.get(obj.category, DEFAULT_OBJECT_COLOR)
    r = max(obj.size / 2.0, 0.035)
    if obj.name in detected:
        shapes.append(Shape("circle", [(obj.x, obj.y)], radius=r + 0.03, outline="#e53935", width=2))
    if obj.name in missed:
        shapes.append(Shape("circle", [(obj.x, obj.y)], radius=r + 0.03, outline="#888888", width=2, dash=True))
    if obj.name == highlight:
        shapes.append(Shape("circle", [(obj.x, obj.y)], radius=r + 0.05, outline="#111111", width=2))
    if obj.category == "Tools":
        shapes.append(Shape("poly", [(obj.x - r, obj.y - r), (obj.x + r, obj.y - r), (obj.x + r, obj.y + r), (obj.x - r, obj.y + r)],
                            fill=color, outline="#ffffff"))
    else:
        shapes.append(Shape("circle", [(obj.x, obj.y)], radius=r, fill=color, outline="#ffffff"))
    if show_labels:
        shapes.append(Shape("text", [(obj.x, obj.y - r - 0.01)], text=labels[obj.name], size=7, fill="#333333", anchor="n"))


def inventory_lines(world: WorldMap, snap: SimSnapshot) -> List[Tuple[str, bool]]:
    '''The side panel: what is really where. (text, is_heading)'''
    lines = []
    places = list(world.locations) + [HELD]
    for place in places:
        objs = sorted(n for n, o in snap.objects.items() if o.location == place)
        if place == HELD and not objs:
            continue
        lines.append((f"{place} ({len(objs)})", True))
        lines += [(f"  {n}", False) for n in objs] or [("  —", False)]
    return lines


def status_lines(snap: SimSnapshot, speed: float, paused: bool) -> List[str]:
    line1 = f"t={snap.time:6.1f}s{'  PAUSED' if paused else ''}  |  carrying: {snap.robot.held or 'nothing'}"
    ops = ", ".join(snap.operations) or "idle"
    line2 = f"ops: {ops}  |  violations: {len(snap.violations)}"
    return [line1, line2]
