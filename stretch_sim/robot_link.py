'''
Connection to the real robot: the behaviour tree runs against the real Stretch, and the
simulator only mirrors what it does on the virtual map, so the viewer and the event log
still show the run.

The mirror is driven by the state the BT publishes (bt_cooking's StretchEnvironment on
/cooking_bt/state): the running node, the target location and object, the last scan and
the held object. The real robot's own topics are not used, and the sim's map is not the
robot's map, so the virtual robot drives to the sim's location of the same name.

    Navigate node running      the virtual robot drives to that location
    scan/objects changes       objects the real robot saw are moved to where it saw them
    Grasp node running         the virtual robot goes to the location and waits: the object
                               is only shown in its gripper once the BT reports the real
                               grasp confirmed (holding_object set), since an animation
                               would finish, and report success, long before the real one
    Handover node running      the virtual robot goes to the Human and waits: the hand over
                               is only shown once the BT reports it done (holding_object
                               cleared, the object believed with the Human), since the
                               real one can fail, e.g. with no /handover service
    holding_object differs     the gripper is corrected to match, whatever the animation
                               did, as soon as no animation is running

The BT navigates with the robot's own location poses, made from its places.yaml (written
by stretch_places), e.g. Table -> table, Human -> handover.
'''
import json
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from stretch_sim.events import INFO
from stretch_sim.geometry import Pose2D, world_to_local
from stretch_sim.perception import Detection
from stretch_sim.simulator import KinematicSimulator, normalise_label
from stretch_sim.world import HELD

DEFAULT_PLACES = "stretch:~/stretch_user/maps/embrace_apartment_map.places.yaml"
DEFAULT_DOMAIN_ID = 42
# BT location -> place in places.yaml, where the place is not the lower-case location name.
DEFAULT_PLACE_NAMES = {"Human": "handover"}
# Degrees added to a place's yaw for a BT location. The arm extends to the robot's right,
# so to hand over it must park with its right side to the person; a place saved facing the
# person (as "handover" was) is turned a quarter turn left, so the arm faces where it faced.
DEFAULT_YAW_OFFSETS_DEG = {"Human": 90.0}
# How far the virtual robot may be from a location's approach pose and still count as there.
AT_LOCATION_M = 0.2


def read_places(source: str, timeout_s: float = 10.0) -> Dict[str, Dict[str, float]]:
    '''
    Places from a stretch_places file: a local path, or HOST:PATH to read it over ssh.
    Returns {place: {"x", "y", "yaw"}} in the robot's map frame.
    '''
    local = Path(source).expanduser()
    if local.is_file():
        text = local.read_text(encoding="utf-8")
    elif ":" in source:
        host, path = source.split(":", 1)
        try:
            proc = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(timeout_s)}", host, "cat", path],
                                  capture_output=True, text=True, timeout=timeout_s + 5.0)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"timed out reading {path} from {host} over ssh")
        if proc.returncode != 0:
            raise RuntimeError(f"cannot read {path} from {host} over ssh: {proc.stderr.strip() or proc.returncode}")
        text = proc.stdout
    else:
        raise RuntimeError(f"no places file at {source}")
    places = (yaml.safe_load(text) or {}).get("places") or {}
    return {name: {"x": float(p["x"]), "y": float(p["y"]), "yaw": float(p["yaw"])} for name, p in places.items()}


def location_poses_from_places(places: Dict[str, Dict[str, float]], locations: List[str],
                               place_names: Optional[Dict[str, str]] = None,
                               yaw_offsets_deg: Optional[Dict[str, float]] = None) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    '''
    The BT's location_poses.json content for the real robot, and the locations that have
    no place. Each location uses place_names[location], else DEFAULT_PLACE_NAMES, else
    its lower-case name, turned by yaw_offsets_deg[location], else DEFAULT_YAW_OFFSETS_DEG.
    '''
    names = DEFAULT_PLACE_NAMES | (place_names or {})
    offsets = DEFAULT_YAW_OFFSETS_DEG | (yaw_offsets_deg or {})
    poses, missing = {}, []
    for location in locations:
        place = names.get(location, location.lower())
        if place in places:
            pose = dict(places[place])
            yaw = pose["yaw"] + math.radians(offsets.get(location, 0.0))
            pose["yaw"] = round(math.atan2(math.sin(yaw), math.cos(yaw)), 4)
            poses[location] = pose
        else:
            missing.append(f"{location} (no place '{place}')")
    return poses, missing


class RobotMirror:
    '''Replays what the BT does with the real robot on the simulator (see the module docstring).'''

    def __init__(self, sim: KinematicSimulator):
        self.sim = sim
        self._tip: Optional[str] = None
        self._scan = None
        self._holding: Optional[str] = None

    def on_bt_state(self, payload: dict) -> None:
        values = payload.get("values") or {}
        tip = payload.get("tip") or ""
        with self.sim.lock:
            if values.get("scan/objects") != self._scan:
                self._scan = values.get("scan/objects")
                self._mirror_scan(values)
            if tip != self._tip:
                self._tip = tip
                self._mirror_node(tip, values)
            if values.get("holding_object") != self._holding:
                previous, self._holding = self._holding, values.get("holding_object")
                self._mirror_holding(previous, self._holding, values)
            elif not self.sim.ops and self.sim.robot.held != self._holding:
                # An animation ended differently from the real robot, e.g. a grasp it failed.
                self._mirror_holding(self._holding, self._holding, values)

    '''
    NODES
    '''
    def _mirror_node(self, tip: str, values: dict) -> None:
        if "Navigate" in tip:
            target = "Human" if "Human" in tip else values.get("target_location")
            if target in self.sim.world.locations:
                self._emit(f"real robot driving to {target}")
                self._cancel_ops()
                self._ensure_mode("navigation")
                self.sim.navigate_to_location(target)
        elif "Grasp" in tip:
            self._mirror_grasp(values)
        elif "Handover" in tip:
            self._arrive("Human")
            if self.sim.robot.held:
                # Not animated: the release is shown when the BT reports the real one done.
                self._emit(f"real robot handing over '{self.sim.robot.held}'; waiting for it to report the result")

    def _mirror_grasp(self, values: dict) -> None:
        name, here = values.get("target_object"), values.get("target_location")
        obj = self.sim.objects.get(name)
        if obj is None or obj.location == HELD or here not in self.sim.world.locations:
            return
        self._arrive(here)
        if obj.location != here:
            self._move(name, here)      # the real robot found it here
        # Not animated: the object goes into the gripper when the BT reports the real grasp done.
        self._emit(f"real robot grasping '{name}' at the {here}; waiting for it to report the result")

    '''
    STATE
    '''
    def _mirror_scan(self, values: dict) -> None:
        scan = self._scan or []
        here = values.get("target_location")
        if here not in self.sim.world.locations or self.sim.world.locations[here].kind != "surface":
            return
        self._arrive(here)
        r = self.sim.robot.pose
        dets, seen = [], []
        for det in scan:
            name = self._object_for(det)
            if name is None:
                continue
            if self.sim.objects[name].location not in (here, HELD):
                self._move(name, here)
            obj = self.sim.objects[name]
            if obj.location == HELD:
                continue
            lx, ly = world_to_local(r, obj.x, obj.y)
            dets.append(Detection(label=det.get("detector_class_name") or det.get("class_name", ""),
                                  score=float(det.get("score", 0.0)), centroid=(lx, ly, obj.z),
                                  size=(obj.size, obj.size, obj.size), stamp=self.sim.time,
                                  grasp_direction=det.get("grasp_direction") or "front",
                                  true_object=name, world_xy=(obj.x, obj.y)))
            seen.append(name)
        self.sim.last_detections = dets
        self.sim.last_scan = {"surface": here, "missed": [], "time": self.sim.time}
        unknown = len(scan) - len(seen)
        self._emit(f"real robot saw {sorted(seen) or 'no known objects'} at the {here}"
                   + (f" and {unknown} detection(s) it has no object for" if unknown else ""),
                   detections=scan)

    def _mirror_holding(self, previous: Optional[str], held: Optional[str], values: dict) -> None:
        robot = self.sim.robot
        if robot.held == held:
            return
        self._cancel_ops()
        if robot.held:
            # Put down what the animation picked up: delivered if the BT now believes the
            # person has it, otherwise back where it was taken from.
            name = robot.held
            delivered = float(values.get(f"confidence/{name}/Human") or 0.0) >= 0.99
            dest = "Human" if delivered else values.get("target_location") or self._nearest_surface()
            robot.held = None
            robot.aperture = 0.0
            self.sim.objects[name].location = dest
            self._move(name, dest)
            self._emit(f"'{name}' {'handed to the Human' if delivered else f'put back on the {dest}'}, as on the real robot")
        if held in self.sim.objects:
            obj = self.sim.objects[held]
            obj.location = HELD
            robot.held = held
            robot.aperture = obj.size
            self._emit(f"real robot is holding '{held}'")

    '''
    HELPERS
    '''
    def _object_for(self, det: dict) -> Optional[str]:
        labels = {normalise_label(det.get("class_name", "")), normalise_label(det.get("detector_class_name", ""))} - {""}
        for name, obj in sorted(self.sim.objects.items()):
            if normalise_label(obj.label) in labels or normalise_label(name) in labels:
                return name
        return None

    def _arrive(self, location: str) -> None:
        '''The real robot is at location, so the virtual one is too, even if it is still driving.'''
        goal = self.sim.world.locations[location].approach
        r = self.sim.robot.pose
        if math.hypot(r.x - goal.x, r.y - goal.y) <= AT_LOCATION_M:
            return
        self._cancel_ops()
        self.sim.robot.pose = Pose2D(goal.x, goal.y, goal.yaw)
        self._emit(f"moved the virtual robot to the {location}, where the real robot is")

    def _move(self, name: str, location: str) -> None:
        try:
            self.sim.move_object(name, location)
        except ValueError as err:
            self._emit(f"could not move '{name}' to the {location}: {err}")

    def _nearest_surface(self) -> str:
        r = self.sim.robot.pose
        surfaces = [l for l in self.sim.world.locations.values() if l.kind == "surface"]
        return min(surfaces, key=lambda l: math.hypot(l.approach.x - r.x, l.approach.y - r.y)).name

    def _ensure_mode(self, mode: str) -> None:
        if self.sim.robot.driver_mode != mode:
            self.sim.set_driver_mode(mode)

    def _cancel_ops(self) -> None:
        for op in list(self.sim.ops):
            self.sim.cancel(op)

    def _emit(self, message: str, **data) -> None:
        self.sim.emit(INFO, "robot", message, **data)


def write_location_poses(poses: Dict[str, Dict[str, float]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(poses, indent=2) + "\n", encoding="utf-8")
    return path
