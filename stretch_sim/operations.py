'''
Long-running robot operations. The simulator steps every active operation each tick;
an operation finishes as SUCCEEDED (the motion completed; check .result for what
happened, e.g. whether anything ended up in the gripper), FAILED (it could not be
carried out) or CANCELED. This mirrors how a ROS action or service would report it.
'''
import math
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

from stretch_sim.geometry import Point, Pose2D, wrap_angle

if TYPE_CHECKING:
    from stretch_sim.simulator import KinematicSimulator


class OpStatus(Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class Operation:
    # Operations that share a resource preempt each other.
    RESOURCES: tuple = ()

    def __init__(self, name: str):
        self.name = name
        self.status = OpStatus.RUNNING
        self.message = ""
        self.result: Dict[str, Any] = {}
        self.elapsed = 0.0
        self.id: Optional[int] = None       # set by the simulator; pairs log events
        self.args: Dict[str, Any] = {}      # the command's arguments, for the log

    @property
    def done(self) -> bool:
        return self.status != OpStatus.RUNNING

    def step(self, sim: "KinematicSimulator", dt: float) -> None:
        raise NotImplementedError

    def succeed(self, message: str = "", **result) -> None:
        self.status, self.message = OpStatus.SUCCEEDED, message
        self.result.update(result)

    def fail(self, message: str, **result) -> None:
        self.status, self.message = OpStatus.FAILED, message
        self.result.update(result)

    def cancel(self, message: str = "canceled") -> None:
        if not self.done:
            self.status, self.message = OpStatus.CANCELED, message

    def describe(self) -> str:
        return f"{self.name} [{self.status.value}] {self.message}".strip()


def _approach(current: float, target: float, max_step: float) -> float:
    delta = target - current
    return target if abs(delta) <= max_step else current + math.copysign(max_step, delta)


class NavigateOp(Operation):
    '''Follows a planned path: turn towards each waypoint, drive to it, then turn to the goal yaw.'''
    RESOURCES = ("base",)

    def __init__(self, goal: Pose2D, path: List[Point], speed: float, turn_rate: float,
                 fail_at_fraction: Optional[float] = None):
        super().__init__("navigate")
        self.goal = goal
        self.path = path
        self.speed = speed
        self.turn_rate = turn_rate
        self.fail_at_fraction = fail_at_fraction
        self._index = 1
        self._total = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
        self._travelled = 0.0

    @property
    def remaining_path(self) -> List[Point]:
        return self.path[self._index:]

    def distance_remaining(self, pose: Pose2D) -> float:
        if self._index >= len(self.path):
            return 0.0
        rest = self.path[self._index:]
        return math.dist((pose.x, pose.y), rest[0]) + sum(math.dist(a, b) for a, b in zip(rest, rest[1:]))

    def step(self, sim, dt):
        pose = sim.robot.pose
        if self._index < len(self.path):
            if self.fail_at_fraction is not None and self._travelled >= self.fail_at_fraction * self._total:
                self.fail("navigation aborted (injected failure)", injected_failure=True)
                return
            wx, wy = self.path[self._index]
            dist = math.hypot(wx - pose.x, wy - pose.y)
            if dist < 1e-3:
                self._waypoint_reached(sim)
                return
            heading = math.atan2(wy - pose.y, wx - pose.x)
            err = wrap_angle(heading - pose.yaw)
            if abs(err) > math.radians(3):
                pose.yaw = wrap_angle(pose.yaw + _approach(0.0, err, self.turn_rate * dt))
                return
            pose.yaw = heading
            move = min(dist, self.speed * dt)
            pose.x += move * math.cos(heading)
            pose.y += move * math.sin(heading)
            self._travelled += move
            if move >= dist:
                self._waypoint_reached(sim)
            return

        err = wrap_angle(self.goal.yaw - pose.yaw)
        if abs(err) > 1e-3:
            pose.yaw = wrap_angle(pose.yaw + _approach(0.0, err, self.turn_rate * dt))
            return
        pose.yaw = self.goal.yaw
        self.succeed("arrived", distance_m=round(self._travelled, 3))

    def _waypoint_reached(self, sim) -> None:
        self._index += 1
        last = self._index >= len(self.path)
        sim.op_step(self, "reached goal position, turning to goal yaw" if last else
                    f"waypoint {self._index - 1}/{len(self.path) - 2} reached",
                    position=[round(sim.robot.pose.x, 3), round(sim.robot.pose.y, 3)])


class PhasedOp(Operation):
    '''
    An operation made of phases run one after another. Each phase is a function
    called every tick with (sim, dt) that returns True once the phase is complete.
    '''
    def __init__(self, name: str):
        super().__init__(name)
        self.phases: List[tuple] = []     # (phase name, fn)
        self.phase_index = 0
        self._announced = -1

    def add(self, name: str, fn: Callable[["KinematicSimulator", float], bool]) -> None:
        self.phases.append((name, fn))

    @property
    def phase_name(self) -> str:
        return self.phases[self.phase_index][0] if self.phase_index < len(self.phases) else "done"

    def step(self, sim, dt):
        while self.phase_index < len(self.phases) and not self.done:
            name, fn = self.phases[self.phase_index]
            if self._announced != self.phase_index:
                self._announced = self.phase_index
                sim.op_step(self, f"phase '{name}'", phase=name)
            if not fn(sim, dt):
                return
            self.phase_index += 1
        if not self.done:
            self.finish(sim)

    def finish(self, sim) -> None:
        self.succeed()

    def describe(self) -> str:
        return f"{self.name}:{self.phase_name} [{self.status.value}] {self.message}".strip()

    # Phase builders
    @staticmethod
    def move_joint(attr: str, target: float, speed: float):
        def fn(sim, dt):
            current = getattr(sim.robot, attr)
            setattr(sim.robot, attr, _approach(current, target, speed * dt))
            return getattr(sim.robot, attr) == target
        return fn

    @staticmethod
    def wait(duration: float):
        state = {"t": 0.0}
        def fn(sim, dt):
            state["t"] += dt
            return state["t"] >= duration
        return fn

    @staticmethod
    def do(action: Callable[["KinematicSimulator"], None]):
        def fn(sim, dt):
            action(sim)
            return True
        return fn


class HeadOp(PhasedOp):
    RESOURCES = ("head",)

    def __init__(self, pan: float, tilt: float, duration: float, start_pan: float, start_tilt: float,
                 fail_at_fraction: Optional[float] = None):
        super().__init__("move_head")
        duration = max(duration, 1e-3)
        pan_speed = max(abs(pan - start_pan) / duration, 1e-6)
        tilt_speed = max(abs(tilt - start_tilt) / duration, 1e-6)

        def both(sim, dt):
            if fail_at_fraction is not None and self.elapsed >= fail_at_fraction * duration:
                self.fail("head trajectory aborted (injected failure)", injected_failure=True)
                return False
            a = PhasedOp.move_joint("head_pan", pan, pan_speed)(sim, dt)
            b = PhasedOp.move_joint("head_tilt", tilt, tilt_speed)(sim, dt)
            return a and b
        self.add("move", both)
