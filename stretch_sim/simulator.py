import math
import random
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from stretch_sim.events import BELIEF, COMMAND, INFO, RESULT, STATE, STEP, VIOLATION, EventLog, SimEvent
from stretch_sim.geometry import Point, Pose2D, world_to_local
from stretch_sim.operations import HeadOp, NavigateOp, Operation, PhasedOp
from stretch_sim.perception import CameraModel, Detection, PerceptionNoise, detect
from stretch_sim.planner import GridPlanner
from stretch_sim.robot import DRIVER_MODES, RobotGeometry, RobotState
from stretch_sim.world import GRASP_DIRECTIONS, HELD, SimObject, WorldMap, free_slot, person_slot, spawn_objects

# Driver modes each command is allowed in. A command in any other mode is recorded as a
# violation and fails, as it would on the robot (e.g. the base ignores velocity commands
# outside navigation mode, so Nav2 never gets anywhere).
DEFAULT_MODE_RULES = {
    "navigate":  ["navigation"],
    "move_head": ["position"],
    "grasp":     ["position"],
    "handover":  ["position"],
}

# How far above the object a top grasp raises the gripper before reaching over it.
TOP_GRASP_CLEARANCE_M = 0.12


def normalise_label(label: str) -> str:
    '''Detector labels compared as the grasp servers do: "Chick peas", "chick_peas" and "chickpeas" match.'''
    return "".join(ch for ch in str(label).lower() if ch.isalnum())


@dataclass
class SimSnapshot:
    '''A consistent copy of the simulation state, safe to read from another thread.'''
    time: float
    robot: RobotState
    objects: Dict[str, SimObject]
    operations: List[str]
    path: List[Point]
    detections: List[Detection]
    violations: List[str]
    events: List[SimEvent]
    scan: Optional[dict] = None        # the last scan, while recent: surface and missed objects


class RobotSimulator(ABC):
    '''
    The interface the viewer and the ROS layer use. The 2D kinematic simulator below
    implements it; a physics-based simulator can replace it without changing either.
    '''
    world: WorldMap
    geom: RobotGeometry
    log: EventLog

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> None: ...
    @abstractmethod
    def step(self, dt: float) -> None: ...
    @abstractmethod
    def snapshot(self) -> SimSnapshot: ...

    @abstractmethod
    def set_driver_mode(self, mode: str) -> Tuple[bool, str]: ...
    @abstractmethod
    def navigate_to(self, goal: Pose2D) -> Operation: ...
    @abstractmethod
    def move_head(self, pan: float, tilt: float, duration: float) -> Operation: ...
    @abstractmethod
    def detect(self) -> List[Detection]: ...
    @abstractmethod
    def grasp(self, point_base: Tuple[float, float, float], direction: str = "front",
              expected_label: str = "") -> Operation: ...
    @abstractmethod
    def handover(self) -> Operation: ...
    @abstractmethod
    def cancel(self, op: Operation) -> None: ...
    @abstractmethod
    def gripper_finger_rad(self) -> float: ...

    # Test and debugging controls: change the world behind the robot's back.
    @abstractmethod
    def shuffle_objects(self, count: int = 3, seed: Optional[int] = None) -> List[Tuple[str, str, str]]:
        '''
        Moves `count` randomly chosen objects to a different surface each, picked uniformly
        rather than by the priors, so the robot's beliefs about them become wrong. Objects
        with a person or in the gripper stay put. Returns (object, from, to) per move.
        '''
        with self.lock:
            rng = random.Random(seed)
            surfaces = [n for n, loc in self.world.locations.items() if loc.kind == "surface"]
            candidates = sorted(n for n, o in self.objects.items() if o.location in surfaces)
            moves = []
            for name in rng.sample(candidates, min(max(count, 0), len(candidates))):
                source = self.objects[name].location
                targets = [s for s in surfaces if s != source]
                rng.shuffle(targets)
                for target in targets:
                    try:
                        self.move_object(name, target)
                    except ValueError:      # that surface is full
                        continue
                    moves.append((name, source, target))
                    break
            self.emit(INFO, "sim", f"shuffled {len(moves)} object(s): " +
                      ", ".join(f"{n} {a} -> {b}" for n, a, b in moves), moves=moves)
            return moves

    def move_object(self, name: str, location: str) -> None: ...


class _FailedOp(Operation):
    def __init__(self, name: str, message: str):
        super().__init__(name)
        self.fail(message)

    def step(self, sim, dt):
        pass


def _detection_summary(d: Detection) -> dict:
    return {"label": d.label, "score": d.score, "centroid": [round(v, 3) for v in d.centroid],
            "grasp_direction": d.grasp_direction, "true_object": d.true_object}


class KinematicSimulator(RobotSimulator):
    '''
    Top-down kinematic simulation: the base follows planned paths at a fixed speed,
    joints move at fixed rates, and perception and grasping are geometric checks with
    configurable failure probabilities. All public methods are thread-safe.

    Everything the robot is asked to do and everything it reports back goes to
    self.log (see stretch_sim.events).
    '''
    def __init__(self, world: WorldMap, scenario: Optional[dict] = None, log: Optional[EventLog] = None):
        self.world = world
        self.scenario = scenario or {}
        self.log = log or EventLog()
        robot_cfg = self.scenario.get("robot", {}) or {}
        self.geom = RobotGeometry(**(robot_cfg.get("geometry") or {}))
        self.camera = CameraModel(**(robot_cfg.get("camera") or {}))
        self.noise = PerceptionNoise.from_config(self.scenario.get("perception"))
        self.nav_cfg = {"speed_mps": 0.3, "turn_rate_rps": 1.0} | (self.scenario.get("navigation") or {})
        self.grasp_cfg = {"match_tolerance_m": 0.06} | (self.scenario.get("grasp") or {})
        self.handover_cfg = {"wait_s": 1.5} | (self.scenario.get("handover") or {})
        # How often each action fails: 1 - success_rate, unless the scenario gives that
        # action its own failure probability.
        self.success_rate = float(robot_cfg.get("success_rate", 1.0))
        self.fail_overrides = {
            "navigate":  self.nav_cfg.get("fail_prob"),
            "move_head": (self.scenario.get("head") or {}).get("fail_prob"),
            "detect":    (self.scenario.get("perception") or {}).get("miss_prob"),    # per object on the surface
            "grasp":     self.grasp_cfg.get("fail_prob"),
            "handover":  self.handover_cfg.get("no_hand_prob"),
        }
        self.mode_rules = DEFAULT_MODE_RULES | (self.scenario.get("driver_mode_rules") or {})
        self.planner = GridPlanner(world, robot_radius=self.geom.radius)
        self.lock = threading.RLock()
        self.time = 0.0
        self._next_op_id = 0
        self.bt_state: Optional[dict] = None    # the behaviour tree's published state, if one is running
        self.reset(self.scenario.get("seed"))

    '''
    LOGGING
    '''
    def emit(self, kind: str, source: str, message: str, op_id: Optional[int] = None, **data) -> None:
        self.log.emit(self.time, kind, source, message, op_id, **data)

    def op_step(self, op: Operation, message: str, **data) -> None:
        '''Called by operations as they progress.'''
        self.emit(STEP, op.name, message, op.id, **data)

    def _register(self, op: Operation, **args) -> Operation:
        self._next_op_id += 1
        op.id = self._next_op_id
        op.args = args
        self.emit(COMMAND, op.name, ", ".join(f"{k}={v}" for k, v in args.items()) or "(no arguments)", op.id, **args)
        return op

    def _report(self, op: Operation) -> None:
        msg = op.status.value + (f": {op.message}" if op.message else "")
        self.emit(RESULT, op.name, msg, op.id, status=op.status.value, duration_s=round(op.elapsed, 2), **op.result)

    def _violation(self, msg: str, op_id: Optional[int] = None) -> None:
        self.violations.append(f"t={self.time:.1f}s: {msg}")
        self.emit(VIOLATION, "rules", msg, op_id)

    '''
    ACTION FAILURES
    '''
    def fail_prob(self, action: str) -> float:
        override = self.fail_overrides.get(action)
        return float(override) if override is not None else 1.0 - self.success_rate

    def failure_probabilities(self) -> Dict[str, float]:
        return {action: round(self.fail_prob(action), 4) for action in self.fail_overrides}

    def _inject_failure(self, action: str) -> bool:
        return self.rng.random() < self.fail_prob(action)

    def set_success_rate(self, rate: float, clear_overrides: bool = True) -> None:
        '''
        Sets how often robot actions succeed, from 0 to 1. By default this also drops
        the scenario's per-action failure probabilities, so the rate applies to every action.
        '''
        with self.lock:
            self.success_rate = min(max(float(rate), 0.0), 1.0)
            if clear_overrides:
                self.fail_overrides = {action: None for action in self.fail_overrides}
            self.emit(INFO, "sim", f"action success rate set to {self.success_rate:.0%}",
                      success_rate=self.success_rate, failure_probabilities=self.failure_probabilities())

    '''
    STATE
    '''
    def reset(self, seed: Optional[int] = None) -> None:
        with self.lock:
            self.seed = seed
            self.rng = random.Random(seed)
            self.objects = spawn_objects(self.world, self.scenario, self.rng)
            robot_cfg = self.scenario.get("robot", {}) or {}
            start = robot_cfg.get("start")
            self.robot = RobotState(
                pose=Pose2D(**start) if start else self.world.robot_start.copy(),
                driver_mode=robot_cfg.get("driver_mode", "position"),
            )
            self.time = 0.0
            self.ops: List[Operation] = []
            self.violations: List[str] = []
            self.last_detections: List[Detection] = []
            self.last_scan: Optional[dict] = None
            placement = {loc: sorted(n for n, o in self.objects.items() if o.location == loc) for loc in self.world.locations}
            self.emit(INFO, "sim", f"reset (seed {seed}): " + ", ".join(f"{loc} {len(v)}" for loc, v in placement.items()),
                      seed=seed, placement=placement, robot=self.robot.pose.to_dict(), driver_mode=self.robot.driver_mode,
                      success_rate=self.success_rate, failure_probabilities=self.failure_probabilities())

    def step(self, dt: float) -> None:
        with self.lock:
            self.time += dt
            for op in list(self.ops):
                if not op.done:
                    op.elapsed += dt
                    op.step(self, dt)
                if op.done:
                    self.ops.remove(op)
                    self._report(op)
            if self.robot.held:
                obj = self.objects[self.robot.held]
                obj.x, obj.y = self.robot.gripper_xy(self.geom)
                obj.z = self.robot.lift

    def snapshot(self) -> SimSnapshot:
        with self.lock:
            path = []
            for op in self.ops:
                if isinstance(op, NavigateOp):
                    path = [(self.robot.pose.x, self.robot.pose.y)] + op.remaining_path
            return SimSnapshot(
                time=self.time,
                robot=self.robot.copy(),
                objects={n: o.copy() for n, o in self.objects.items()},
                operations=[f"#{op.id} {op.describe()}" for op in self.ops],
                path=path,
                detections=[d for d in self.last_detections if self.time - d.stamp < 3.0],
                scan=self.last_scan if self.last_scan and self.time - self.last_scan["time"] < 3.0 else None,
                violations=list(self.violations),
                events=self.log.events[-10:],
            )

    def location_of(self, name: str) -> str:
        with self.lock:
            return self.objects[name].location

    '''
    COMMANDS
    '''
    def _start(self, op: Operation) -> Operation:
        for other in list(self.ops):
            if set(other.RESOURCES) & set(op.RESOURCES):
                other.cancel(f"preempted by #{op.id} {op.name}")
                self.ops.remove(other)
                self._report(other)
        self.ops.append(op)
        return op

    def _reject(self, op: Operation, msg: str) -> Operation:
        '''Finish a just-registered command immediately as failed.'''
        failed = _FailedOp(op.name, msg)
        failed.id, failed.args = op.id, op.args
        self._report(failed)
        return failed

    def _check_mode(self, op: Operation) -> Optional[Operation]:
        allowed = self.mode_rules.get(op.name)
        if allowed and self.robot.driver_mode not in allowed:
            msg = f"{op.name} requested in '{self.robot.driver_mode}' mode (needs {' or '.join(allowed)})"
            self._violation(msg, op.id)
            return self._reject(op, msg)
        return None

    def cancel(self, op: Operation) -> None:
        with self.lock:
            if op in self.ops:
                op.cancel()
                self.ops.remove(op)
                self._report(op)

    def set_driver_mode(self, mode: str) -> Tuple[bool, str]:
        with self.lock:
            self.emit(COMMAND, "set_driver_mode", f"mode={mode}", mode=mode)
            if mode not in DRIVER_MODES:
                msg = f"unknown mode '{mode}'"
                self.emit(RESULT, "set_driver_mode", f"failed: {msg}", success=False)
                return False, msg
            if mode != self.robot.driver_mode:
                # Switching modes stops the base, as the driver does.
                for op in list(self.ops):
                    if "base" in op.RESOURCES:
                        op.cancel(f"driver switched to {mode} mode")
                        self.ops.remove(op)
                        self._report(op)
                self.emit(STATE, "driver", f"mode {self.robot.driver_mode} -> {mode}",
                          previous=self.robot.driver_mode, mode=mode)
                self.robot.driver_mode = mode
            msg = f"Now in {mode} mode."
            self.emit(RESULT, "set_driver_mode", f"succeeded: {msg}", success=True)
            return True, msg

    def navigate_to(self, goal: Pose2D) -> Operation:
        with self.lock:
            target = self.world.location_at_pose(goal)
            op = self._register(NavigateOp(goal, [], 0.0, 0.0), x=round(goal.x, 3), y=round(goal.y, 3),
                                yaw=round(goal.yaw, 3), location=target)
            failed = self._check_mode(op)
            if failed:
                return failed
            path = self.planner.plan((self.robot.pose.x, self.robot.pose.y), (goal.x, goal.y))
            if path is None:
                return self._reject(op, f"no path to ({goal.x:.2f}, {goal.y:.2f})")
            fail_at = self.rng.uniform(0.2, 0.8) if self._inject_failure("navigate") else None
            nav = NavigateOp(goal, path, float(self.nav_cfg["speed_mps"]), float(self.nav_cfg["turn_rate_rps"]), fail_at)
            nav.id, nav.args = op.id, op.args
            self.op_step(nav, f"planned path with {len(path) - 2} intermediate waypoint(s), "
                              f"{sum(math.dist(a, b) for a, b in zip(path, path[1:])):.2f} m",
                         path=[[round(x, 3), round(y, 3)] for x, y in path])
            return self._start(nav)

    def navigate_to_location(self, name: str) -> Operation:
        return self.navigate_to(self.world.locations[name].approach.copy())

    def move_head(self, pan: float, tilt: float, duration: float = 2.0) -> Operation:
        with self.lock:
            fail_at = self.rng.uniform(0.2, 0.8) if self._inject_failure("move_head") else None
            op = self._register(HeadOp(pan, tilt, duration, self.robot.head_pan, self.robot.head_tilt, fail_at),
                                pan=round(pan, 3), tilt=round(tilt, 3), duration_s=duration)
            failed = self._check_mode(op)
            if failed:
                return failed
            return self._start(op)

    def detect(self) -> List[Detection]:
        with self.lock:
            r = self.robot
            self._next_op_id += 1
            op_id = self._next_op_id
            self.emit(COMMAND, "detect", f"pose ({r.pose.x:.2f}, {r.pose.y:.2f}, yaw {r.pose.yaw:.2f})", op_id,
                      pose=r.pose.to_dict())
            scan = detect(self.world, self.camera, r.pose, self.objects.values(), self.noise, self.rng,
                          self.time, self.fail_prob("detect"))
            dets = scan.detections
            self.last_detections = dets
            self.last_scan = {"surface": scan.surface, "missed": scan.missed, "time": self.time}
            if scan.surface is None:
                message = "0 detection(s): the robot is not at a surface"
            else:
                message = f"{len(dets)} detection(s) on the {scan.surface}: {[d.label for d in dets]}"
                if scan.missed:
                    message += f"; missed {scan.missed} (injected failure)"
            self.emit(RESULT, "detect", message, op_id, surface=scan.surface,
                      detections=[_detection_summary(d) for d in dets], missed=scan.missed,
                      injected_failure=bool(scan.missed))
            return dets

    def gripper_finger_rad(self) -> float:
        with self.lock:
            return self.robot.finger_joint_rad(self.geom)

    def grasp(self, point_base: Tuple[float, float, float], direction: str = "front",
              expected_label: str = "") -> Operation:
        '''
        Grasp at a point in the base frame. A front grasp, as the grasp_carry server does:
        open, shift the base along its heading to line the arm up, lift, extend, close,
        raise, retract. A top grasp, as topdown_grasp does, raises the gripper above the
        object first, extends over it and lowers onto it before closing.

        The point is interpreted in the base frame *now*, so a detection from before the
        base last moved misses. Succeeds once the motion is done; whether something is
        held is in result["held"] and in the gripper aperture. With expected_label, the
        held object's detector label is checked against it, as the servers' grasp check
        does: result["wrong_object"] is set when they differ.
        '''
        with self.lock:
            g, r = self.geom, self.robot
            lx, ly, z = (float(v) for v in point_base)
            args = {"point_base": [round(lx, 3), round(ly, 3), round(z, 3)], "direction": direction}
            if expected_label:
                args["expected_label"] = expected_label
            op = self._register(PhasedOp("grasp"), **args)
            failed = self._check_mode(op)
            if failed:
                return failed
            if direction not in GRASP_DIRECTIONS:
                return self._reject(op, f"unknown grasp direction '{direction}' (expected one of {list(GRASP_DIRECTIONS)})")
            reach = -ly
            if r.held:
                return self._reject(op, f"already holding '{r.held}'")
            if not g.arm_min_reach - 0.05 <= reach <= g.arm_max_reach:
                return self._reject(op, f"target {reach:.2f} m to the side is out of the arm's reach "
                                        f"({g.arm_min_reach:.2f}-{g.arm_max_reach:.2f} m)")
            if abs(lx) > g.max_align_m:
                return self._reject(op, f"target {lx:.2f} m along the base is too far to line up")
            if not g.lift_min <= z <= g.lift_max:
                return self._reject(op, f"target height {z:.2f} m is outside the lift range")

            extension = min(max(reach - g.arm_min_reach, 0.0), g.arm_max_extension)
            align = {"left": lx}

            def shift_base(sim, dt):
                step = math.copysign(min(abs(align["left"]), 0.15 * dt), align["left"])
                p = sim.robot.pose
                p.x += step * math.cos(p.yaw)
                p.y += step * math.sin(p.yaw)
                align["left"] -= step
                return abs(align["left"]) < 1e-6

            close = {"target": None}

            def close_gripper(sim, dt):
                if close["target"] is None:
                    obj = sim._object_at_gripper()
                    if obj is not None and sim._inject_failure("grasp"):
                        sim.op_step(op, f"'{obj.name}' slipped (injected failure)", object=obj.name)
                        op.result["injected_failure"] = True
                        op.result["slipped"] = obj.name
                        obj = None
                    elif obj is None:
                        gx, gy = sim.robot.gripper_xy(sim.geom)
                        sim.op_step(op, "nothing at the gripper", gripper=[round(gx, 3), round(gy, 3), round(sim.robot.lift, 3)])
                    else:
                        sim.op_step(op, f"closing on '{obj.name}'", object=obj.name)
                    close["obj"] = obj
                    close["target"] = obj.size if obj else 0.0
                done = PhasedOp.move_joint("aperture", close["target"], 0.08)(sim, dt)
                if done and close["obj"] is not None:
                    close["obj"].location = HELD
                    sim.robot.held = close["obj"].name
                return done

            op.add("open", PhasedOp.move_joint("aperture", g.gripper_open_m, 0.1))
            op.add("align", shift_base)
            if direction == "top":
                # Clear the object before reaching over it, then come down onto it.
                op.add("lift above", PhasedOp.move_joint("lift", min(z + TOP_GRASP_CLEARANCE_M, g.lift_max), 0.15))
                op.add("extend", PhasedOp.move_joint("arm_extension", extension, 0.12))
                op.add("lower", PhasedOp.move_joint("lift", z, 0.1))
            else:
                op.add("lift", PhasedOp.move_joint("lift", z, 0.15))
                op.add("extend", PhasedOp.move_joint("arm_extension", extension, 0.12))
            op.add("close", close_gripper)
            op.add("raise", PhasedOp.move_joint("lift", min(z + 0.08, g.lift_max), 0.1))
            op.add("retract", PhasedOp.move_joint("arm_extension", 0.0, 0.15))

            def finish(sim):
                held = sim.robot.held
                result = {"held": held, "aperture_m": round(sim.robot.aperture, 4)}
                if held:
                    message = f"holding '{held}'"
                    label = sim.objects[held].label
                    if expected_label:
                        result["wrong_object"] = normalise_label(label) != normalise_label(expected_label)
                        if result["wrong_object"]:
                            message += f", detected as '{label}', not the expected '{expected_label}'"
                elif op.result.get("slipped"):
                    message = f"'{op.result['slipped']}' slipped out, gripper empty (injected failure)"
                else:
                    message = "closed on nothing"
                op.succeed(message, **result)

            op.finish = finish
            return self._start(op)

    def _object_at_gripper(self) -> Optional[SimObject]:
        gx, gy = self.robot.gripper_xy(self.geom)
        tol = float(self.grasp_cfg["match_tolerance_m"])
        candidates = [o for o in self.objects.values() if o.location != HELD
                      and math.hypot(o.x - gx, o.y - gy) <= tol and abs(o.z - self.robot.lift) <= 0.08]
        return min(candidates, key=lambda o: math.hypot(o.x - gx, o.y - gy), default=None)

    def handover(self) -> Operation:
        '''
        Presents the held object to the nearest person on the arm side, waits for a hand,
        releases and retracts. result["handed_over"] says whether the person took it.
        '''
        with self.lock:
            g, r = self.geom, self.robot
            op = self._register(PhasedOp("handover"), holding=r.held)
            failed = self._check_mode(op)
            if failed:
                return failed
            if not r.held:
                return self._reject(op, "nothing in the gripper")

            reachable = []
            for person in (loc for loc in self.world.locations.values() if loc.kind == "person"):
                lx, ly = world_to_local(r.pose, person.rect.cx, person.rect.cy)
                if ly < 0 and abs(lx) < 0.5 and -ly <= g.arm_max_reach + 0.5:
                    reachable.append((math.hypot(lx, ly), person, ly))
            if not reachable:
                return self._reject(op, "no person within reach on the arm side")
            _, person, ly = min(reachable, key=lambda t: t[0])

            extension = min(max(-ly - 0.35 - g.arm_min_reach, 0.0), g.arm_max_extension)
            outcome = {}

            def wait_for_hand(sim, dt):
                if "hand" not in outcome:
                    outcome["hand"] = not sim._inject_failure("handover")
                    if not outcome["hand"]:
                        op.result["injected_failure"] = True
                    outcome["t"] = 0.0
                outcome["t"] += dt
                done = outcome["t"] >= float(sim.handover_cfg["wait_s"])
                if done:
                    sim.op_step(op, "hand detected" if outcome["hand"] else "no hand (injected failure)")
                return done

            def release(sim, dt):
                if not outcome["hand"]:
                    return True
                done = PhasedOp.move_joint("aperture", g.gripper_open_m, 0.1)(sim, dt)
                if done and sim.robot.held:
                    obj = sim.objects[sim.robot.held]
                    index = sum(o.location == person.name for o in sim.objects.values())
                    obj.location = person.name
                    obj.x, obj.y = person_slot(person, index)
                    obj.z = person.height
                    sim.robot.held = None
                    sim.op_step(op, f"'{obj.name}' released to {person.name}", object=obj.name)
                return done

            op.add("lift", PhasedOp.move_joint("lift", min(max(person.height, g.lift_min), g.lift_max), 0.15))
            op.add("extend", PhasedOp.move_joint("arm_extension", extension, 0.12))
            op.add("wait_for_hand", wait_for_hand)
            op.add("release", release)
            op.add("retract", PhasedOp.move_joint("arm_extension", 0.0, 0.15))
            op.finish = lambda sim: op.succeed("handed over" if outcome["hand"] else "no hand detected, still holding",
                                               handed_over=outcome["hand"], person=person.name)
            return self._start(op)

    '''
    BEHAVIOUR TREE STATE
    '''
    def set_bt_state(self, payload: dict) -> None:
        '''
        Records the state a behaviour tree published (see bt_cooking's StretchEnvironment)
        and logs what changed in its beliefs: location probabilities, target and held object.
        '''
        with self.lock:
            old = (self.bt_state or {}).get("values", {})
            new = payload.get("values", {})
            self.bt_state = payload

            changes = {}
            for key in ("target_object", "target_location", "holding_object", "recipe/name"):
                if old.get(key) != new.get(key):
                    changes[key] = [old.get(key), new.get(key)]
            per_object = {}
            for key, value in new.items():
                parts = key.split("/")
                if len(parts) != 3 or parts[0] != "confidence" or parts[1] in ("human_reachable", "required"):
                    continue
                before = old.get(key)
                if before is None or abs(float(before) - float(value)) >= 0.005:
                    per_object.setdefault(parts[1], {})[parts[2]] = [before, value]

            if changes:
                self.emit(BELIEF, "bt", "; ".join(f"{k}: {a} -> {b}" for k, (a, b) in changes.items()), changes=changes)
            if per_object and old:      # the first message is the prior, not a change
                for obj, locs in sorted(per_object.items()):
                    text = ", ".join(f"{loc} {float(a or 0):.0%} -> {float(b):.0%}" for loc, (a, b) in sorted(locs.items()))
                    self.emit(BELIEF, "bt", f"{obj}: {text}", object=obj, changes=locs)
            elif per_object:
                self.emit(BELIEF, "bt", f"received beliefs about {len(per_object)} objects from '{payload.get('tree')}'")

    '''
    TEST AND DEBUG CONTROLS
    '''
    def shuffle_objects(self, count: int = 3, seed: Optional[int] = None) -> List[Tuple[str, str, str]]:
        '''
        Moves `count` randomly chosen objects to a different surface each, picked uniformly
        rather than by the priors, so the robot's beliefs about them become wrong. Objects
        with a person or in the gripper stay put. Returns (object, from, to) per move.
        '''
        with self.lock:
            rng = random.Random(seed)
            surfaces = [n for n, loc in self.world.locations.items() if loc.kind == "surface"]
            candidates = sorted(n for n, o in self.objects.items() if o.location in surfaces)
            moves = []
            for name in rng.sample(candidates, min(max(count, 0), len(candidates))):
                source = self.objects[name].location
                targets = [s for s in surfaces if s != source]
                rng.shuffle(targets)
                for target in targets:
                    try:
                        self.move_object(name, target)
                    except ValueError:      # that surface is full
                        continue
                    moves.append((name, source, target))
                    break
            self.emit(INFO, "sim", f"shuffled {len(moves)} object(s): " +
                      ", ".join(f"{n} {a} -> {b}" for n, a, b in moves), moves=moves)
            return moves

    def move_object(self, name: str, location: str) -> None:
        with self.lock:
            obj = self.objects[name]
            if obj.location == HELD:
                raise ValueError(f"'{name}' is in the gripper")
            previous = obj.location
            loc = self.world.locations[location]
            if loc.kind == "surface":
                obj.x, obj.y = free_slot(self.world, {n: o for n, o in self.objects.items() if n != name}, location)
                obj.z = loc.height + obj.size / 2.0
            else:
                obj.x, obj.y = person_slot(loc, sum(o.location == location for o in self.objects.values() if o.name != name))
                obj.z = loc.height
            obj.location = location
            self.emit(INFO, "sim", f"moved '{name}' from {previous} to {location}", object=name, previous=previous, location=location)
