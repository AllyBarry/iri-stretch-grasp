'''
Runs the simulation in real time on a background thread and executes manual commands
from a user interface. UI-independent: the web viewer uses it, and so can anything else
that wants to drive the simulation by hand.
'''
import math
import random
import threading
import time
from typing import Any, Dict, List, Optional

from stretch_sim.bt_runner import discover_recipes, discover_trees
from stretch_sim.events import INFO
from stretch_sim.operations import NavigateOp, PhasedOp
from stretch_sim.script import ScriptRunner, fetch, tour
from stretch_sim.simulator import KinematicSimulator
from stretch_sim.world import HELD

SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


class CommandError(Exception):
    pass


class SimController:
    # Commands that wait on something outside the simulation, so must not hold its lock.
    UNLOCKED_COMMANDS = ("reset_memory", "connect", "disconnect", "bt_stop", "cancel", "release_runstop")
    # Commands that act as the robot. In robot mode the virtual robot only mirrors the real one.
    ROBOT_COMMANDS = ("goto", "detect", "grasp_best", "handover", "mode", "demo", "tour", "success_rate")

    def __init__(self, sim: KinematicSimulator, speed: float = 1.0, scenario_name: str = "",
                 demo_object: Optional[str] = None, step_s: float = 0.05, link=None, robot_mode: bool = False):
        self.sim = sim
        self.link = link                # the RosLink, which switches between the sim and the real robot
        self._robot_mode = robot_mode   # without a link, e.g. in tests
        self.speed = speed
        self.paused = False
        self.scenario_name = scenario_name
        self.demo_object = demo_object
        self.step_s = step_s
        self.runner = ScriptRunner(sim)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def robot_mode(self) -> bool:
        '''The BT runs on the real robot and the sim mirrors it (robot_link.py).'''
        return self.link.mode == "robot" if self.link is not None else self._robot_mode

    @property
    def bt(self):
        '''The BtProcess, when ROS is running.'''
        return self.link.bt if self.link is not None and self.link.mode is not None else None

    @property
    def ros_bridge(self):
        return self.link.bridge if self.link is not None else None

    @property
    def ros_status(self) -> Dict[str, Any]:
        return self.link.status() if self.link is not None else {"available": False, "reason": "started without --ros"}

    '''
    SIMULATION LOOP
    '''
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="sim-loop", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self.link is not None:
            self.link.stop()
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self.runner.stop()

    def _loop(self) -> None:
        last = time.monotonic()
        while not self._stop.is_set():
            time.sleep(self.step_s / 2.0)
            now = time.monotonic()
            wall_dt, last = min(now - last, 0.25), now
            if self.paused:
                continue
            remaining = wall_dt * self.speed
            while remaining > 1e-9:
                dt = min(remaining, self.step_s)
                with self.sim.lock:
                    self.sim.step(dt)
                    self.runner.update()
                remaining -= dt

    '''
    STATE FOR THE UI
    '''
    def state(self) -> Dict[str, Any]:
        sim = self.sim
        with sim.lock:
            snap = sim.snapshot()
            r = snap.robot
            gx, gy = r.gripper_xy(sim.geom)
            ops = [self._op_state(op) for op in sim.ops]
            script = self.runner.name if self.runner.active else None
        detected = {d.true_object for d in snap.detections}
        return {
            "time": round(snap.time, 2),
            "speed": self.speed,
            "paused": self.paused,
            "seed": sim.seed,
            "scenario": self.scenario_name,
            "robot": {
                "x": r.pose.x, "y": r.pose.y, "yaw": r.pose.yaw,
                "mode": r.driver_mode, "pan": r.head_pan, "tilt": r.head_tilt,
                "lift": r.lift, "arm": r.arm_extension, "aperture": r.aperture,
                "held": r.held, "gripper": [gx, gy],
            },
            "scan": snap.scan,
            "path": [[round(x, 3), round(y, 3)] for x, y in snap.path],
            "objects": [{"name": o.name, "category": o.category, "location": o.location, "label": o.label,
                         "grasp_direction": o.grasp_direction, "x": round(o.x, 3), "y": round(o.y, 3), "z": round(o.z, 3), "size": o.size,
                         "detected": o.name in detected} for o in snap.objects.values()],
            "operations": ops,
            "script": script,
            "success_rate": sim.success_rate,
            "failure_probabilities": sim.failure_probabilities(),
            "bt": self.bt.status() if self.bt is not None else None,
            "robot_mode": self.robot_mode,
            "ros": self.ros_status,
            "beliefs": self._beliefs(),
            "violations": snap.violations,
        }

    def _beliefs(self) -> Optional[Dict[str, Any]]:
        '''What the behaviour tree believes, from the state it publishes, or None.'''
        payload = self.sim.bt_state
        if not payload:
            return None
        values = payload.get("values", {})
        probs: Dict[str, Dict[str, float]] = {}
        for key, value in values.items():
            parts = key.split("/")
            if len(parts) == 3 and parts[0] == "confidence" and parts[1] not in ("human_reachable", "required"):
                probs.setdefault(parts[1], {})[parts[2]] = round(float(value or 0.0), 4)
        world_order = list(self.sim.world.locations)
        seen = {loc for locs in probs.values() for loc in locs}
        locations = [l for l in world_order if l in seen] + sorted(seen - set(world_order))
        return {
            "tree": payload.get("tree"),
            "status": payload.get("status"),
            "tip": payload.get("tip"),
            "recipe": values.get("recipe/name"),
            "target_object": values.get("target_object"),
            "target_location": values.get("target_location"),
            "holding_object": values.get("holding_object"),
            "locations": locations,
            "probabilities": probs,
            "required": sorted(o for o in probs if values.get(f"required/{o}")),
            "human_reachable": {o: bool(values.get(f"human_reachable/{o}")) for o in probs},
        }

    def _op_state(self, op) -> Dict[str, Any]:
        state = {"id": op.id, "name": op.name, "status": op.status.value, "message": op.message,
                 "args": op.args, "elapsed": round(op.elapsed, 1), "phase": None, "progress": None}
        if isinstance(op, PhasedOp) and op.phases:
            state["phase"] = op.phase_name
            state["progress"] = op.phase_index / len(op.phases)
        elif isinstance(op, NavigateOp):
            total = sum(math.dist(a, b) for a, b in zip(op.path, op.path[1:])) or 1.0
            remaining = op.distance_remaining(self.sim.robot.pose)
            state["phase"] = f"{remaining:.1f} m to go" if remaining > 0 else "turning"
            state["progress"] = 1.0 - remaining / total
        return state

    def world(self) -> Dict[str, Any]:
        w, g = self.sim.world, self.sim.geom
        return {
            "name": w.name,
            "width": w.width,
            "height": w.height,
            "walls": [[list(a), list(b)] for a, b in w.walls],
            "locations": [{"name": l.name, "kind": l.kind, "color": l.color, "height": l.height,
                           "rect": {"cx": l.rect.cx, "cy": l.rect.cy, "w": l.rect.w, "d": l.rect.d},
                           "approach": l.approach.to_dict()} for l in w.locations.values()],
            "obstacles": [{"name": n, "rect": {"cx": r.cx, "cy": r.cy, "w": r.w, "d": r.d}, "color": w.obstacle_colors.get(n)}
                          for n, r in w.obstacles],
            "robot": {"base_size": g.base_size, "arm_min_reach": g.arm_min_reach,
                      "arm_max_extension": g.arm_max_extension, "lift_min": g.lift_min, "lift_max": g.lift_max,
                      "gripper_open_m": g.gripper_open_m},
            "speeds": SPEEDS,
            "ros": self.ros_status,
            "bt_trees": discover_trees(),
            "recipes": discover_recipes(),
        }

    '''
    COMMANDS
    '''
    def execute(self, cmd: Dict[str, Any]) -> str:
        name = cmd.get("cmd")
        handler = getattr(self, f"_cmd_{name}", None)
        if handler is None:
            raise CommandError(f"unknown command '{name}'")
        if self.robot_mode and name in self.ROBOT_COMMANDS:
            raise CommandError(f"'{name}' is not available while mirroring the real robot; run a behaviour tree instead")
        if name in self.UNLOCKED_COMMANDS:
            return handler(cmd) or "ok"
        with self.sim.lock:
            return handler(cmd) or "ok"

    def _ensure_mode(self, mode: str) -> None:
        '''Manual commands switch to the driver mode they need first, so they just work.'''
        if self.sim.robot.driver_mode != mode:
            self.sim.set_driver_mode(mode)

    def _cmd_goto(self, cmd):
        location = cmd.get("location")
        if location not in self.sim.world.locations:
            raise CommandError(f"unknown location '{location}'")
        self._ensure_mode("navigation")
        self.sim.navigate_to_location(location)
        return f"driving to {location}"

    def _cmd_detect(self, cmd):
        return f"{len(self.sim.detect())} detection(s)"

    def _cmd_grasp_best(self, cmd):
        detections = self.sim.snapshot().detections
        if not detections:
            raise CommandError("no recent detections to grasp; detect first")
        best = max(detections, key=lambda d: d.score)
        self._ensure_mode("position")
        self.sim.grasp(best.centroid, direction=best.grasp_direction, expected_label=best.label)
        return f"grasping the '{best.label}' detection from the {best.grasp_direction}"

    def _cmd_handover(self, cmd):
        self._ensure_mode("position")
        self.sim.handover()

    def _cmd_mode(self, cmd):
        mode = cmd.get("mode") or ("navigation" if self.sim.robot.driver_mode == "position" else "position")
        ok, msg = self.sim.set_driver_mode(mode)
        if not ok:
            raise CommandError(msg)
        return msg

    def _cancel_sim(self) -> None:
        '''Stops the virtual robot: scripts and every running operation.'''
        with self.sim.lock:
            self.runner.stop()
            for op in list(self.sim.ops):
                self.sim.cancel(op)

    def _cmd_cancel(self, cmd):
        '''
        Stops everything. With the real robot, that also stops the tree and runstops the
        robot, since a grasp, handover or scan service part way has no other stop.
        '''
        if not self.robot_mode:
            self._cancel_sim()
            return "canceled everything"
        ok, message = self._stop_real_robot(runstop=True)
        if not ok:
            raise CommandError(f"the robot may still be moving: {message}")
        return "runstopped the robot and stopped the tree; release the runstop to move it again"

    def _stop_real_robot(self, runstop: bool):
        '''
        Stops the robot at once, then the tree, then cancels any goal the tree sent while
        it was shutting down. Returns (ok, message) for the robot stop.
        '''
        ok, message = self.ros_bridge.stop_robot(runstop=runstop)
        bt, bridge = self.bt, self.ros_bridge
        if bt is not None and bt.running:
            def stop_tree():
                bt.stop()
                bridge.cancel_robot_goals()
            threading.Thread(target=stop_tree, daemon=True).start()   # may take a few seconds
        self._cancel_sim()
        return ok, message

    def _cmd_release_runstop(self, cmd):
        if not self.robot_mode:
            raise CommandError("not connected to the robot")
        ok, message = self.ros_bridge.release_runstop()
        if not ok:
            raise CommandError(f"runstop not released: {message}")
        return "released the runstop"

    def _graspable_objects(self) -> List[str]:
        return [n for n, o in self.sim.objects.items()
                if o.location != HELD and self.sim.world.locations[o.location].kind == "surface"]

    def _cmd_demo(self, cmd):
        candidates = self._graspable_objects()
        if not candidates:
            raise CommandError("no objects on surfaces")
        obj = cmd.get("object") or (self.demo_object if self.demo_object in candidates else random.choice(candidates))
        if obj not in candidates:
            raise CommandError(f"'{obj}' is not on a surface")
        self.runner.start(f"fetch {obj}", fetch(self.sim, obj))
        return f"fetching {obj}"

    def _cmd_tour(self, cmd):
        self.runner.start("tour", tour(self.sim))

    def _cmd_pause(self, cmd):
        self.paused = bool(cmd.get("paused", not self.paused))
        self.sim.emit(INFO, "viewer", "paused" if self.paused else "resumed")
        return "paused" if self.paused else "running"

    def _cmd_speed(self, cmd):
        self.speed = min(max(float(cmd.get("speed", 1.0)), SPEEDS[0]), SPEEDS[-1])
        return f"speed x{self.speed:g}"

    def _cmd_success_rate(self, cmd):
        try:
            rate = float(cmd["rate"])
        except (KeyError, TypeError, ValueError):
            raise CommandError("success_rate needs a numeric 'rate' between 0 and 1")
        if not 0.0 <= rate <= 1.0:
            raise CommandError("rate must be between 0 and 1")
        self.sim.set_success_rate(rate)
        return f"actions now succeed {self.sim.success_rate:.0%} of the time"

    def _cmd_bt_start(self, cmd):
        if self.bt is None:
            raise CommandError(f"behaviour trees need the ROS interface: {self.ros_status.get('reason', 'not running')}")
        tree = cmd.get("tree")
        if tree not in discover_trees():
            raise CommandError(f"unknown tree '{tree}'")
        recipe = cmd.get("recipe") or None
        if recipe and recipe not in discover_recipes():
            raise CommandError(f"unknown recipe '{recipe}'")
        try:
            self.bt.start(tree, recipe, int(cmd.get("max_passes", 10)))
        except (RuntimeError, OSError) as err:
            raise CommandError(str(err))
        return f"started '{tree}'"

    def _cmd_bt_stop(self, cmd):
        '''
        Stops the tree and what it set going: the virtual robot's operations, or with the
        real robot, its goals and current motion (even with no tree running, e.g. after it
        crashed). A grasp, handover or scan service part way on the robot carries on; Cancel
        everything runstops it.
        '''
        if self.robot_mode:
            ok, message = self._stop_real_robot(runstop=False)
            if not ok:
                raise CommandError(f"the robot may still be moving: {message}")
            return f"stopped the robot ({message})"
        if self.bt is None or not self.bt.running:
            raise CommandError("no behaviour tree is running")
        bt = self.bt

        def stop_tree():
            bt.stop()
            self._cancel_sim()      # what the tree started would otherwise run on
        threading.Thread(target=stop_tree, daemon=True).start()   # may take a few seconds
        return "stopping"

    def _cmd_reset_memory(self, cmd):
        if self.ros_bridge is None or self.bt is None or not self.bt.running:
            raise CommandError("no behaviour tree is running")
        self.sim.emit(INFO, "viewer", "asked the behaviour tree to reset its memory")
        # Not under the simulation lock: the tree answers between its own ticks.
        ok, message = self.ros_bridge.reset_bt_memory()
        self.sim.emit(INFO, "bt", f"memory reset {'done' if ok else 'failed'}: {message}")
        if not ok:
            raise CommandError(message)
        return message

    def _cmd_connect(self, cmd):
        '''Drive the real robot instead of the sim; places and domain_id default to the last ones used.'''
        if self.link is None:
            raise CommandError("the simulator was started without its ROS link")
        domain = cmd.get("domain_id")
        try:
            self.link.start_robot(places=(cmd.get("places") or "").strip() or None,
                                  domain_id=int(domain) if domain not in (None, "") else None)
        except (RuntimeError, ValueError) as err:
            raise CommandError(f"not connected: {err}")
        except ImportError as err:
            raise CommandError(f"not connected: ROS is not available ({err}); start the simulator from a sourced ROS 2 environment")
        robot = self.link.robot
        return (f"connected to the robot on ROS domain {robot['domain_id']}"
                + (f"; no place for {', '.join(robot['missing'])}" if robot["missing"] else ""))

    def _cmd_disconnect(self, cmd):
        '''Back to the sim playing the robot.'''
        if self.link is None:
            raise CommandError("the simulator was started without its ROS link")
        try:
            self.link.start_sim()
        except ImportError as err:
            raise CommandError(f"ROS is not available ({err})")
        return "disconnected: the sim plays the robot again"

    def _cmd_respawn(self, cmd):
        self._cancel_sim()
        seed = cmd.get("seed")
        self.sim.reset(int(seed) if seed not in (None, "") else random.randrange(1_000_000))
        return f"respawned with seed {self.sim.seed}"

    def _cmd_shuffle(self, cmd):
        try:
            count = int(cmd.get("count", 3))
        except (TypeError, ValueError):
            raise CommandError("shuffle needs a whole number 'count'")
        if not 1 <= count <= 50:
            raise CommandError("count must be between 1 and 50")
        moves = self.sim.shuffle_objects(count)
        if not moves:
            raise CommandError("no objects on surfaces to shuffle")
        return "moved " + ", ".join(f"{n} ({a} \u2192 {b})" for n, a, b in moves)

    def _cmd_move_object(self, cmd):
        name, location = cmd.get("name"), cmd.get("location")
        if name not in self.sim.objects or location not in self.sim.world.locations:
            raise CommandError(f"cannot move '{name}' to '{location}'")
        try:
            self.sim.move_object(name, location)
        except ValueError as err:
            raise CommandError(str(err))
        return f"moved {name} to {location}"
