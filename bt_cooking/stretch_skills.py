import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import py_trees
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionClient

from action_msgs.msg import GoalStatus
from control_msgs.action import FollowJointTrajectory
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

from stretch_floor_perception.srv import GraspPoint, ScanSurface

from unaceba.bt.nodes import ActionNode, ConditionNode
from unaceba.bt.rosnodes import RosActionNode, RosConditionNode

from bt_cooking.stretch_state import StretchState

Status = py_trees.common.Status

'''
DECISION-MAKING
'''
class CalculateReachabilityNode(ActionNode):
    # "Human" is where an object ends up after a successful handover.
    _ACCESSIBLE_LOCATIONS = ["Fridge","Table","Human"]

    def __init__(self, name:str = "CalculateReachabilityNode", reachable_threshold: float = 0.5):
        super(CalculateReachabilityNode, self).__init__(name)

        self.reachable_threshold = reachable_threshold
        self.board.register_key("environment", access=py_trees.common.Access.WRITE)

    def execute(self, state:StretchState):
        current_recipe = state.get_value("recipe/name")
        recipe_data = StretchState.load_recipe(current_recipe)

        required = set(recipe_data.get("ingredients", [])) | set(recipe_data.get("tools", []))

        for obj in sorted(StretchState.get_objects()):
            # Locations are mutually exclusive, so probabilities add.
            prob = sum(state.get_value(f"confidence/{obj}/{loc}") for loc in self._ACCESSIBLE_LOCATIONS)
            prob = min(prob, 1.0)  # guard against rounding / non-normalised data

            state.set_value(f"human_reachable/{obj}",prob > self.reachable_threshold)
            state.set_value(f"confidence/human_reachable/{obj}",prob)
            state.set_value(f"required/{obj}",obj in required)
            state.set_value(f"confidence/required/{obj}",1.0) # TODO: What about underspecified recipes?

        return Status.SUCCESS

    def input_variables(self):
        location_confs = [f"confidence/{obj}/{loc}" for obj in StretchState.get_objects() for loc in self._ACCESSIBLE_LOCATIONS]


        return location_confs + ["recipe/name"]

    def output_variables(self):
        objects = StretchState.get_objects()

        reachables = [f"human_reachable/{obj}" for obj in objects]
        reachable_confs = [f"confidence/human_reachable/{obj}" for obj in objects]
        requireds = [f"required/{obj}" for obj in objects]
        required_confs = [f"confidence/required/{obj}" for obj in objects]

        return reachables + reachable_confs + requireds + required_confs

    def semantic_description(self):
        return "Calculates the human reachability of each object and whether or not they are required by the recipe known to the robot"

class DecideTargetObjectNode(ActionNode):
    def __init__(self, name:str = "DecideTargetObjectNode"):
        super(DecideTargetObjectNode, self).__init__(name)

        self.board.register_key("environment", access=py_trees.common.Access.WRITE)

    def execute(self, state:StretchState):
        current_recipe = state.get_value("recipe/name")
        recipe_data = StretchState.load_recipe(current_recipe)

        ordered_objects = self.get_objects_by_first_appearance(recipe_data)

        blocking_objects = [obj for obj in StretchState.get_objects() if state.get_value(f"required/{obj}") and not state.get_value(f"human_reachable/{obj}")]

        blocking = set(blocking_objects)
        next_obj = next((obj for obj in ordered_objects if obj in blocking), None)

        state.set_value("target_object",next_obj)

        return Status.SUCCESS


    def input_variables(self):
        known_objects = StretchState.get_objects()
        requireds = [f"required/{obj}" for obj in known_objects]
        reachables = [f"human_reachable/{obj}" for obj in known_objects]
        return requireds + reachables + ["recipe/name"]

    def output_variables(self):
        return ["target_object"]

    def semantic_description(self):
        return "Based on which objects are reachable by the human, which are required by the recipe, and the order of steps in the recipe, decides the next object to fetch"


    @staticmethod
    def get_objects_by_first_appearance(recipe_data):
        known_objects = StretchState.get_objects()
        arg_keys = ("object", "tool", "source", "target")

        seen = set()
        ordered = [] 

        for step in recipe_data.get("steps", []):
            for action in step.get("actions", []):
                args = action.get("args", {})
                for key in arg_keys:
                    value = args.get(key)
                    if value in known_objects and value not in seen:
                        seen.add(value)
                        ordered.append(value)

        return ordered

class DecideTargetLocationNode(ActionNode):
    '''
    Picks the most likely location of the target object as the next place to search.
    SearchForObjectNode lowers a location's confidence when the object is not seen there,
    so repeated failures move the search on to the next most likely location.
    '''
    def __init__(self, name: str = "DecideTargetLocationNode", excluded_locations: tuple = ("Human",)):
        super().__init__(name)

        self.excluded_locations = set(excluded_locations)

    def _search_locations(self):
        return [loc for loc in StretchState.get_locations() if loc not in self.excluded_locations]

    def execute(self, state: StretchState):
        obj = state.get_value("target_object")
        if obj is None:
            state.set_value("target_location", None)
            return Status.FAILURE

        probs = {loc: state.get_value(f"confidence/{obj}/{loc}") for loc in self._search_locations()}
        best = max(probs, key=probs.get, default=None)
        if best is None or probs[best] <= 0.0:
            state.set_value("target_location", None)
            return Status.FAILURE

        state.set_value("target_location", best)
        return Status.SUCCESS

    def input_variables(self):
        confs = [f"confidence/{obj}/{loc}" for obj in StretchState.get_objects() for loc in self._search_locations()]
        return confs + ["target_object"]

    def output_variables(self):
        return ["target_location"]

    def semantic_description(self):
        return "Decides where to search for the target object, choosing the location where it is most likely to be"

'''
EXECUTING ROBOT SKILLS

The runner spins the shared ROS node once per tick, so none of these nodes block:
they start a service call or action goal, return RUNNING, and poll the result on later ticks.
'''
class _SkillLogging:
    def _info(self, msg: str) -> None:
        self._ros_node.get_logger().info(f"[{self.name}] {msg}")

    def _warn(self, msg: str) -> None:
        self._ros_node.get_logger().warn(f"[{self.name}] {msg}")

    def _error(self, msg: str) -> None:
        self._ros_node.get_logger().error(f"[{self.name}] {msg}")


# How long a skill waits for the robot's service or action server to appear. Over the
# lab WiFi a fresh BT process takes 7-12 s to discover the robot (multicast from the
# laptop does not reach it, so the robot only learns of the BT once the BT has heard the
# robot's periodic announcement), so a few seconds is not enough for the first skill.
SERVER_TIMEOUT_S = 30.0
# How long a service may be gone while a skill waits for its response before the skill
# gives up: a server that crashes part way (e.g. scan_surface) never answers.
SERVER_GONE_S = 5.0


class _DriverModeSwitch:
    '''
    Puts stretch_driver in the mode a skill needs before the skill sends its request:
    Nav2 needs "navigation" mode (base velocity control), and the grasp and handover
    skills move the arm, which needs "position" mode. This is part
    of each skill rather than a separate step in the tree. Like the skills, it never
    blocks: step() is called once per tick.
    '''
    MODES = ("position", "navigation", "trajectory")

    def __init__(self, mode: str, timeout_s: float):
        if mode not in self.MODES:
            raise ValueError(f"Unknown driver mode '{mode}', expected one of {self.MODES}")
        self.mode         = mode
        self.service_name = f"/switch_to_{mode}_mode"
        self.timeout_s    = timeout_s
        self._client      = None
        self.reset()

    def setup(self, ros_node) -> None:
        self._client = ros_node.create_client(Trigger, self.service_name)

    def reset(self) -> None:
        self._start_s = time.monotonic()
        self._future  = None
        self.done     = False

    def step(self, skill) -> Optional[Status]:
        '''None once the driver is in the mode, RUNNING while switching, FAILURE if it cannot switch.'''
        if self.done:
            return None
        if self._future is None:
            if not self._client.service_is_ready():
                if time.monotonic() - self._start_s > self.timeout_s:
                    skill._error(f"Driver service '{self.service_name}' not available after {self.timeout_s:.1f}s.")
                    return Status.FAILURE
                return Status.RUNNING
            self._future = self._client.call_async(Trigger.Request())
            return Status.RUNNING
        if not self._future.done():
            return Status.RUNNING

        exception = self._future.exception()
        response = None if exception is not None else self._future.result()
        if response is None or not response.success:
            reason = exception or (response.message if response is not None else "no response")
            skill._error(f"Could not switch the driver to {self.mode} mode: {reason}")
            return Status.FAILURE
        self.done = True
        return None

    def cancel(self) -> None:
        if self._future is not None and not self._future.done():
            self._client.remove_pending_request(self._future)
        self.reset()


class _GripperStow:
    '''
    Readies the arm for driving, in position mode. With an empty gripper it stows as the
    robot's "stow" alias does: wrist yaw 3.4, pitch -0.5, roll 0 and the gripper closed,
    lift and arm left where they are. Carrying an object (set carrying before the first
    step), it only draws the arm back: twisting the wrist inwards would swing the object
    into the robot's mast. A step a skill runs first, like _DriverModeSwitch; step() is
    called once per tick and never blocks. It goes by the goal's status, so a driver on
    another ROS distribution (whose results may be unreadable) is fine.
    '''
    POSE = {"joint_wrist_yaw": 3.4, "joint_wrist_pitch": -0.5, "joint_wrist_roll": 0.0,
            "joint_gripper_finger_left": 0.0}
    CARRY_POSE = {"wrist_extension": 0.0}

    def __init__(self, action_name: str = "/stretch_controller/follow_joint_trajectory",
                 timeout_s: float = SERVER_TIMEOUT_S + 15.0, move_s: float = 3.0):
        self.action_name = action_name
        self.timeout_s   = timeout_s
        self.move_s      = move_s
        self._mode       = _DriverModeSwitch("position", SERVER_TIMEOUT_S)
        self._client     = None
        self.reset()

    def setup(self, ros_node) -> None:
        self._mode.setup(ros_node)
        self._client = ActionClient(ros_node, FollowJointTrajectory, self.action_name)

    def reset(self) -> None:
        self._mode.reset()
        self._start_s     = time.monotonic()
        self._goal_future = None
        self._handle      = None
        self.done         = False
        self.carrying     = False

    def step(self, skill) -> Optional[Status]:
        '''None once stowed, RUNNING while stowing, FAILURE if it cannot stow.'''
        if self.done:
            return None
        if time.monotonic() - self._start_s > self.timeout_s:
            skill._error(f"Stowing the gripper did not finish within {self.timeout_s:.0f} s.")
            self.cancel()
            return Status.FAILURE
        if self._goal_future is None:
            status = self._mode.step(skill)
            if status is not None:
                return status
            if not self._client.server_is_ready():
                return Status.RUNNING
            pose = self.CARRY_POSE if self.carrying else self.POSE
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = list(pose)
            point = JointTrajectoryPoint(positions=[float(v) for v in pose.values()])
            point.time_from_start.sec = int(self.move_s)
            goal.trajectory.points = [point]
            skill._info("Draw the arm back to carry the object before navigating." if self.carrying
                        else "Stow the gripper before navigating.")
            self._goal_future = self._client.send_goal_async(goal)
            return Status.RUNNING
        if self._handle is None:
            if not self._goal_future.done():
                return Status.RUNNING
            handle = None if self._goal_future.exception() is not None else self._goal_future.result()
            if handle is None or not handle.accepted:
                skill._error("The driver rejected the gripper stow goal.")
                return Status.FAILURE
            self._handle = handle
            return Status.RUNNING
        status = self._handle.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.done = True
            return None
        if status in (GoalStatus.STATUS_ABORTED, GoalStatus.STATUS_CANCELED):
            skill._error(f"Stowing the gripper failed (goal status {status}).")
            return Status.FAILURE
        return Status.RUNNING

    def cancel(self) -> None:
        if self._handle is not None and self._handle.status not in (
                GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED):
            self._handle.cancel_goal_async()
        self._mode.cancel()
        self.reset()


class RosServiceNode(_SkillLogging, RosActionNode):
    '''
    Base class for a skill that is a single ROS service call.

    Subclasses set SRV_TYPE and implement make_request() and on_response(). A subclass
    that sets DRIVER_MODE switches the driver to that mode before calling the service.
    Service calls cannot be cancelled, so a preempted or timed-out call keeps
    running on the server; its response is discarded. If the server disappears while
    the call is pending (it crashed), the node fails after SERVER_GONE_S rather than
    waiting for a response that will never come.
    '''
    SRV_TYPE = None
    DRIVER_MODE: Optional[str] = None

    def __init__(self, name: str, service_name: str, server_timeout_s: float = SERVER_TIMEOUT_S,
                 response_timeout_s: Optional[float] = None):
        super().__init__(name)
        self.service_name       = service_name
        self.server_timeout_s   = server_timeout_s
        self.response_timeout_s = response_timeout_s
        self._client            = None
        self._mode_switch       = _DriverModeSwitch(self.DRIVER_MODE, server_timeout_s) if self.DRIVER_MODE else None
        self._reset()

    def _reset(self) -> None:
        self._start_s  = time.monotonic()
        self._sent_s   = None
        self._future   = None
        self._response = None
        self._gone_s   = None       # since when the service has been gone while a call is pending
        if self._mode_switch is not None:
            self._mode_switch.reset()

    def ros_setup(self) -> None:
        self._client = self._ros_node.create_client(self.SRV_TYPE, self.service_name)
        if self._mode_switch is not None:
            self._mode_switch.setup(self._ros_node)

    def initialise(self) -> None:
        self._reset()

    def make_request(self, state: StretchState):
        '''
        Returns the request to send, None to wait another tick, or a Status to finish without calling the service.
        '''
        raise NotImplementedError

    def on_response(self, state: StretchState, response) -> Status:
        '''
        Handles the response. Called on every tick after the response arrives until it returns something other than RUNNING.
        '''
        raise NotImplementedError

    def execute(self, state: StretchState) -> Status:
        if self._response is not None:
            return self.on_response(state, self._response)

        if self._future is None and self._mode_switch is not None:
            status = self._mode_switch.step(self)
            if status is not None:
                return status

        if self._future is None:
            if not self._client.service_is_ready():
                if time.monotonic() - self._start_s > self.server_timeout_s:
                    self._error(f"Service '{self.service_name}' not available after {self.server_timeout_s:.1f}s.")
                    return Status.FAILURE
                return Status.RUNNING

            request = self.make_request(state)
            if request is None:
                return Status.RUNNING
            if isinstance(request, Status):
                return request

            self._future = self._client.call_async(request)
            self._sent_s = time.monotonic()
            return Status.RUNNING

        if not self._future.done():
            if self.response_timeout_s is not None and time.monotonic() - self._sent_s > self.response_timeout_s:
                self._error(f"No response from '{self.service_name}' after {self.response_timeout_s:.1f}s.")
                return Status.FAILURE
            if self._client.service_is_ready():
                self._gone_s = None
            elif self._gone_s is None:
                self._gone_s = time.monotonic()
            elif time.monotonic() - self._gone_s > SERVER_GONE_S:
                self._error(f"'{self.service_name}' went away before answering (did its node crash?).")
                self._client.remove_pending_request(self._future)
                self._future = None
                return Status.FAILURE
            return Status.RUNNING

        exception = self._future.exception()
        if exception is not None:
            self._error(f"Call to '{self.service_name}' failed: {exception}")
            return Status.FAILURE

        response = self._future.result()
        if response is None:
            self._error(f"Call to '{self.service_name}' returned no response.")
            return Status.FAILURE

        self._response = response
        return self.on_response(state, response)

    def terminate(self, new_status) -> None:
        if self._future is not None and not self._future.done():
            self._warn(
                f"Stopped ({new_status}) before '{self.service_name}' responded. "
                f"The server keeps running the request; its response will be discarded."
            )
            self._client.remove_pending_request(self._future)
        if self._mode_switch is not None:
            self._mode_switch.cancel()
        self._reset()


class RosActionClientNode(_SkillLogging, RosActionNode):
    '''
    Base class for a skill that is a single ROS action goal.

    Subclasses set ACTION_TYPE and implement make_goal() and on_result(). A subclass
    that sets DRIVER_MODE switches the driver to that mode before sending the goal.
    The goal is cancelled if the node is preempted or times out.

    The goal's outcome normally comes with its result. But a server on another ROS
    distribution can send a result this one cannot read, e.g. Nav2's NavigateToPose,
    whose result is empty on Humble (the robot) but has error fields on Jazzy; reading
    it fails with a "Fast CDR exception" and the result never arrives. So:
      - a server whose types carry no type hash (Humble or older, while this is newer)
        is not asked for its result: the node finishes on the goal status from the
        action's status topic, with result None;
      - otherwise, if the goal has finished but no result arrives within
        result_grace_s, the node also finishes on the status.
    '''
    ACTION_TYPE = None
    DRIVER_MODE: Optional[str] = None
    TERMINAL_STATUSES = (GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED, GoalStatus.STATUS_ABORTED)

    def __init__(self, name: str, action_name: str, server_timeout_s: float = SERVER_TIMEOUT_S,
                 result_timeout_s: Optional[float] = None, result_grace_s: float = 2.0):
        super().__init__(name)
        self.action_name      = action_name
        self.server_timeout_s = server_timeout_s
        self.result_timeout_s = result_timeout_s
        self.result_grace_s   = result_grace_s
        self._client          = None
        self._mode_switch     = _DriverModeSwitch(self.DRIVER_MODE, server_timeout_s) if self.DRIVER_MODE else None
        self._reset()

    def _reset(self) -> None:
        self._start_s       = time.monotonic()
        self._sent_s        = None
        self._goal_future   = None
        self._goal_handle   = None
        self._result_future = None
        self._finished_s    = None      # when the status topic first showed the goal finished
        self._status_only   = False     # the server's results cannot be read, so go by the goal status
        if self._mode_switch is not None:
            self._mode_switch.reset()

    def ros_setup(self) -> None:
        self._client = ActionClient(self._ros_node, self.ACTION_TYPE, self.action_name)
        if self._mode_switch is not None:
            self._mode_switch.setup(self._ros_node)

    def initialise(self) -> None:
        self._reset()

    def make_goal(self, state: StretchState):
        '''
        Returns the goal to send, None to wait another tick, or a Status to finish without sending a goal.
        '''
        raise NotImplementedError

    def on_result(self, state: StretchState, goal_status: int, result) -> Status:
        raise NotImplementedError

    def execute(self, state: StretchState) -> Status:
        if self._goal_future is None and self._mode_switch is not None:
            status = self._mode_switch.step(self)
            if status is not None:
                return status

        if self._goal_future is None:
            if not self._client.server_is_ready():
                if time.monotonic() - self._start_s > self.server_timeout_s:
                    self._error(f"Action server '{self.action_name}' not available after {self.server_timeout_s:.1f}s.")
                    return Status.FAILURE
                return Status.RUNNING

            goal = self.make_goal(state)
            if goal is None:
                return Status.RUNNING
            if isinstance(goal, Status):
                return goal

            self._goal_future = self._client.send_goal_async(goal)
            self._sent_s = time.monotonic()
            return Status.RUNNING

        if self._timed_out():
            self._error(f"No result from '{self.action_name}' after {self.result_timeout_s:.1f}s.")
            return Status.FAILURE

        if self._goal_handle is None:
            if not self._goal_future.done():
                return Status.RUNNING
            exception = self._goal_future.exception()
            if exception is not None:
                self._error(f"Sending goal to '{self.action_name}' failed: {exception}")
                return Status.FAILURE
            goal_handle = self._goal_future.result()
            if not goal_handle.accepted:
                self._error(f"Goal rejected by '{self.action_name}'.")
                return Status.FAILURE
            self._goal_handle = goal_handle
            if self._server_lacks_type_hashes():
                self._status_only = True
                self._info(f"'{self.action_name}' runs on an older ROS distribution, so going by its goal status "
                           f"rather than its result.")
            else:
                self._result_future = goal_handle.get_result_async()
            return Status.RUNNING

        if self._result_future is not None and self._result_future.done():
            wrapped = self._result_future.result()
            return self.on_result(state, wrapped.status, wrapped.result)

        # The client keeps the handle's status up to date from the action's status topic.
        goal_status = self._goal_handle.status
        if goal_status not in self.TERMINAL_STATUSES:
            return Status.RUNNING
        if self._status_only:
            return self.on_result(state, goal_status, None)
        if self._finished_s is None:
            self._finished_s = time.monotonic()
        if time.monotonic() - self._finished_s < self.result_grace_s:
            return Status.RUNNING
        self._warn(f"'{self.action_name}' finished (goal status {goal_status}) but its result did not arrive within "
                   f"{self.result_grace_s:.0f} s (a server on another ROS distribution, or slow ticks); "
                   f"going by the status.")
        self._result_future = None      # nothing left to cancel
        return self.on_result(state, goal_status, None)

    def _timed_out(self) -> bool:
        return self.result_timeout_s is not None and time.monotonic() - self._sent_s > self.result_timeout_s

    def _server_lacks_type_hashes(self) -> bool:
        '''
        Whether the action server publishes its types without type hashes, which ROS
        distributions before Iron (e.g. the robot's Humble) do, while this one has them.
        False when unknown, e.g. when this distribution has no type hashes either.
        '''
        try:
            infos = self._ros_node.get_publishers_info_by_topic(f"{self.action_name}/_action/status")
        except Exception:
            return False
        hashes = [getattr(info, "topic_type_hash", None) for info in infos]
        if not hashes or any(h is None for h in hashes):
            return False
        return all(getattr(h, "version", -1) <= 0 for h in hashes)

    def terminate(self, new_status) -> None:
        goal_running = (self._goal_handle is not None and self._goal_handle.status not in self.TERMINAL_STATUSES
                        and (self._status_only or (self._result_future is not None and not self._result_future.done())))
        if goal_running:
            self._warn(f"Stopped ({new_status}) before '{self.action_name}' finished, so cancelling the goal.")
            self._goal_handle.cancel_goal_async()
        elif self._goal_future is not None and not self._goal_future.done():
            # Not accepted yet: cancel as soon as it is.
            self._warn(f"Stopped ({new_status}) before '{self.action_name}' accepted the goal, so cancelling it once accepted.")
            self._goal_future.add_done_callback(self._cancel_when_accepted)
        if self._mode_switch is not None:
            self._mode_switch.cancel()
        self._reset()

    @staticmethod
    def _cancel_when_accepted(future) -> None:
        if future.exception() is None and future.result().accepted:
            future.result().cancel_goal_async()


class _GripperMonitor:
    '''
    Tracks the gripper aperture from /stretch/joint_states. The driver publishes the
    finger angle, which hello_helpers' GripperConversion maps to aperture as
    aperture = 2 * finger_rad * finger_length. The grasp servers squeeze the gripper
    to aperture 0, so an aperture clearly above 0 means the fingers stopped on an object.
    '''
    FINGER_JOINT    = "joint_gripper_finger_left"
    FINGER_LENGTH_M = 0.171

    def __init__(self, topic: str):
        self.topic       = topic
        self._sub        = None
        self.aperture_m  = None
        self.positions: Dict[str, float] = {}   # every joint in the last message, e.g. joint_lift, wrist_extension

    def start(self, ros_node) -> None:
        self.aperture_m = None
        self.positions = {}
        if self._sub is None:
            self._sub = ros_node.create_subscription(JointState, self.topic, self._on_joint_states, 1)

    def stop(self, ros_node) -> None:
        if self._sub is not None:
            ros_node.destroy_subscription(self._sub)
            self._sub = None

    def _on_joint_states(self, msg: JointState) -> None:
        self.positions = dict(zip(msg.name, msg.position))
        if self.FINGER_JOINT in msg.name:
            finger_rad = msg.position[msg.name.index(self.FINGER_JOINT)]
            self.aperture_m = 2.0 * finger_rad * self.FINGER_LENGTH_M


class _TfLookup:
    '''
    Transforms from TF, for skills that need to know where things are. The listener has
    a node of its own, spun on its own thread: the runner spins the BT's node once per
    tick, far too slowly for /tf. One per ROS context, shared by every skill.
    '''
    _shared: Dict[int, "_TfLookup"] = {}

    @classmethod
    def for_node(cls, ros_node) -> "_TfLookup":
        key = id(ros_node.context)
        if key not in cls._shared:
            cls._shared[key] = cls(ros_node)
        return cls._shared[key]

    def __init__(self, ros_node):
        from tf2_ros import Buffer, TransformListener
        self.node = rclpy.create_node(f"{ros_node.get_name()}_tf", context=ros_node.context)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self.node, spin_thread=True)

    def lookup(self, target: str, source: str):
        '''The latest transform taking points in source to target, or None if TF does not have it (yet).'''
        try:
            return self.buffer.lookup_transform(target, source, rclpy.time.Time())
        except Exception:
            return None

    @staticmethod
    def apply(transform, point) -> List[float]:
        '''point ([x, y, z] in the transform's source frame) in its target frame.'''
        t, q = transform.transform.translation, transform.transform.rotation
        x, y, z = (float(v) for v in point)
        # v' = v + 2w(u x v) + 2u x (u x v), with u the quaternion's vector part
        cx, cy, cz = q.y * z - q.z * y, q.z * x - q.x * z, q.x * y - q.y * x
        ccx, ccy, ccz = q.y * cz - q.z * cy, q.z * cx - q.x * cz, q.x * cy - q.y * cx
        return [x + 2.0 * (q.w * cx + ccx) + t.x, y + 2.0 * (q.w * cy + ccy) + t.y, z + 2.0 * (q.w * cz + ccz) + t.z]


def load_location_poses() -> Dict[str, Dict[str, float]]:
    '''
    Loads config/locations/location_poses.json, mapping each location to a map-frame pose:
        {"Fridge": {"x": 1.2, "y": -0.4, "yaw": 1.57}, "Human": {...}, ...}
    The STRETCH_LOCATION_POSES environment variable (set by run_cooking_task's
    --location-poses) points to a different file, e.g. for another map.
    '''
    override = os.environ.get("STRETCH_LOCATION_POSES")
    path = Path(override) if override else \
        Path(get_package_share_directory("stretch_floor_perception")) / "config" / "locations" / "location_poses.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class NavigateToPlaceNode(RosActionClientNode):
    '''
    Drives to a named location with Nav2's NavigateToPose. The location is fixed at
    construction (see NavigateToHumanNode) or read from "target_location".
    First readies the arm (_GripperStow; stow_first): stowed as the robot's "stow" alias
    does, or, carrying an object ("gripper_occupied"), only drawn back. Then switches the
    driver to navigation mode. If that first step fails, so does the drive.
    '''
    ACTION_TYPE = NavigateToPose
    DRIVER_MODE = "navigation"

    def __init__(self, name: str = "NavigateToPlaceNode", location: Optional[str] = None,
                 location_poses: Optional[Dict[str, Dict[str, float]]] = None,
                 action_name: str = "/navigate_to_pose", map_frame: str = "map", stow_first: bool = True, **kwargs):
        super().__init__(name, action_name, **kwargs)
        self.location       = location
        self.location_poses = location_poses
        self.map_frame      = map_frame
        self._stow          = _GripperStow() if stow_first else None

    def initialise(self) -> None:
        super().initialise()
        if self._stow is not None:
            self._stow.reset()

    def execute(self, state):
        if self._stow is not None and not self._stow.done:
            if self._stow._goal_future is None:
                self._stow.carrying = bool(state.get_value("gripper_occupied"))
            status = self._stow.step(self)
            if status is not None:
                return status
        return super().execute(state)

    def terminate(self, new_status) -> None:
        if self._stow is not None and not self._stow.done:
            self._stow.cancel()
        super().terminate(new_status)

    def ros_setup(self) -> None:
        super().ros_setup()
        if self._stow is not None:
            self._stow.setup(self._ros_node)
        if self.location_poses is None:
            self.location_poses = load_location_poses()
        if not self.location_poses:
            self._warn("No location poses configured (config/locations/location_poses.json), so navigation will fail.")

    def _target(self, state):
        return self.location if self.location is not None else state.get_value("target_location")

    def make_goal(self, state):
        location = self._target(state)
        if location is None:
            self._error("No target location set.")
            return Status.FAILURE
        if location not in self.location_poses:
            self._error(f"No pose configured for location '{location}'.")
            return Status.FAILURE

        pose = self.location_poses[location]
        yaw  = float(pose.get("yaw", 0.0))

        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id    = self.map_frame
        goal.pose.header.stamp       = self._ros_node.get_clock().now().to_msg()
        goal.pose.pose.position.x    = float(pose["x"])
        goal.pose.pose.position.y    = float(pose["y"])
        goal.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self._info(f"Navigating to '{location}' at ({goal.pose.pose.position.x:.2f}, {goal.pose.pose.position.y:.2f}, yaw={yaw:.2f}).")
        return goal

    def on_result(self, state, goal_status, result):
        location = self._target(state)
        if goal_status == GoalStatus.STATUS_SUCCEEDED:
            self._info(f"Arrived at '{location}'.")
            return Status.SUCCESS
        self._warn(f"Navigation to '{location}' failed (goal status {goal_status}, error {getattr(result, 'error_code', '?')}).")
        return Status.FAILURE

    def input_variables(self) -> List[str]:
        stow = ["gripper_occupied"] if self._stow is not None else []
        return stow + ([] if self.location is not None else ["target_location"])

    def output_variables(self) -> List[str]:
        return []

    def semantic_description(self):
        target = f"the {self.location} location" if self.location is not None else "the target location"
        return f"Navigates the robot to {target}"


class NavigateToHumanNode(NavigateToPlaceNode):
    '''
    Drives to the "Human" location, where the human waits for the object to be handed over.
    '''
    def __init__(self, name: str = "NavigateToHumanNode", **kwargs):
        super().__init__(name, location="Human", **kwargs)

    def semantic_description(self):
        return "Navigates the robot to the human, to hand over the object it is carrying"


class ScanSurfaceNode(RosServiceNode):
    '''
    Scans the surface at "target_location" with the scan_surface service: the head steps
    over the surface defined for it on the robot (its surfaces.yaml), detecting at each
    pose, and the views are merged. Writes every object found on the surface to
    "scan/objects" as a list of plain dicts (the state is JSON-logged every tick, so it
    cannot hold ROS messages).

    A location's surface is surfaces[location] if given, else its name in lower case
    with underscores (Pantry -> pantry). With no target location, the service's fixed
    head poses are used and nothing is filtered. Head goals are accepted in navigation
    mode, so the driver mode is left as it is. Centroids are in base_link, so they are
    only valid until the base moves.
    '''
    SRV_TYPE = ScanSurface

    def __init__(self, name: str = "ScanSurfaceNode", service_name: str = "/scan_surface",
                 surfaces: Optional[Dict[str, str]] = None, min_score: float = 0.0, **kwargs):
        super().__init__(name, service_name, **kwargs)
        self.surfaces  = surfaces or {}
        self.min_score = min_score

    def surface_for(self, location: Optional[str]) -> str:
        if not location:
            return ""
        return self.surfaces.get(location, "_".join(location.strip().lower().split()))

    def make_request(self, state):
        request = ScanSurface.Request()
        request.surface_name = self.surface_for(state.get_value("target_location"))
        # No target_label: a full sweep, so every object on the surface is reported.
        self._info(f"Scanning surface '{request.surface_name}'." if request.surface_name
                   else "Scanning from the fixed head poses.")
        return request

    def on_response(self, state, response):
        if not response.success:
            # The sweep could not run (unknown surface, head motion failed, no images): no scan, no beliefs.
            self._warn(f"Scan failed: '{response.message}'")
            return Status.FAILURE
        detections = [self._to_dict(det) for det in response.detections.detections if det.score >= self.min_score]
        state.set_value("scan/objects", detections)
        self._info(f"Detected {len(detections)} object(s) in {response.views_scanned} view(s): "
                   f"{[det['class_name'] for det in detections]}")
        return self.on_detections(state, detections)

    def on_detections(self, state, detections: List[dict]) -> Status:
        return Status.SUCCESS

    @staticmethod
    def _to_dict(det) -> dict:
        return {
            "class_name":          det.class_name,
            "detector_class_name": det.detector_class_name,
            "pick_up":             det.pick_up,
            "grasp_direction":     det.grasp_direction,
            "score":               float(det.score),
            "frame_id":            det.header.frame_id,
            "stamp":               {"sec": det.header.stamp.sec, "nanosec": det.header.stamp.nanosec},
            "centroid":            [det.centroid.x, det.centroid.y, det.centroid.z],
            # The 3D box, in frame_id. Its x axis is the object's long axis, which the grasp servers align the fingers with.
            "bbox_center":         [det.bbox3d.center.position.x, det.bbox3d.center.position.y, det.bbox3d.center.position.z],
            "bbox_orientation":    [det.bbox3d.center.orientation.x, det.bbox3d.center.orientation.y,
                                    det.bbox3d.center.orientation.z, det.bbox3d.center.orientation.w],
            "size":                [det.bbox3d.size.x, det.bbox3d.size.y, det.bbox3d.size.z],
        }

    def input_variables(self):
        return ["target_location"]

    def output_variables(self):
        return ["scan/objects"]

    def semantic_description(self):
        return "Sweeps the camera over the surface at the target location and lists the objects on it"


class SearchForObjectNode(ScanSurfaceNode):
    '''
    Scans the surface and looks for "target_object" among the detections. On success the
    best match is written to "target_detection" for GraspObjectNode.

    Also updates the belief about where the target is: seeing it at "target_location"
    makes that location certain; missing it there scales that location's probability by
    (1 - detection_prob) and renormalises, since the detector can miss objects.

    Every other known object seen in the scan is also now believed to be at
    "target_location". Objects that are not seen are left as they were: the detector
    can miss them. The object in the gripper is never moved. A scan that could not run
    (e.g. no surface defined for the location) fails without changing any belief.

    Detector labels are matched case-insensitively as whole words against the object
    name, or against the names in aliases, e.g. {"Olive Oil": ["olive oil", "bottle"]}.
    '''
    def __init__(self, name: str = "SearchForObjectNode", aliases: Optional[Dict[str, List[str]]] = None,
                 detection_prob: float = 0.9, **kwargs):
        super().__init__(name, **kwargs)
        self.aliases        = aliases or {}
        self.detection_prob = detection_prob

    def make_request(self, state):
        if state.get_value("target_object") is None:
            self._error("No target object to search for.")
            return Status.FAILURE
        return super().make_request(state)

    def _matches(self, obj: str, det: dict) -> bool:
        labels = [det["class_name"].strip().lower(), det["detector_class_name"].strip().lower()]
        for alias in self.aliases.get(obj, [obj]):
            pattern = rf"\b{re.escape(alias.strip().lower())}\b"
            if any(re.search(pattern, label) for label in labels):
                return True
        return False

    def on_detections(self, state, detections):
        obj = state.get_value("target_object")
        matches = [det for det in detections if self._matches(obj, det)]
        best = max(matches, key=lambda det: det["score"], default=None)

        state.set_value("target_detection", best)
        state.set_value("target_found", best is not None)
        self._update_location_belief(state, obj, found=best is not None)
        self._update_other_objects(state, obj, detections)

        if best is None:
            self._warn(f"'{obj}' not found.")
            return Status.FAILURE
        self._info(f"Found '{obj}' as '{best['class_name']}' (score {best['score']:.2f}) at {best['centroid']}.")
        return Status.SUCCESS

    def _update_location_belief(self, state, obj: str, found: bool) -> None:
        searched = state.get_value("target_location")
        if searched is None:
            return

        locations = StretchState.get_locations()
        if found:
            probs = {loc: float(loc == searched) for loc in locations}
        else:
            probs = {loc: state.get_value(f"confidence/{obj}/{loc}") for loc in locations}
            probs[searched] *= (1.0 - self.detection_prob)
            total = sum(probs.values())
            if total <= 0.0:
                return
            probs = {loc: p / total for loc, p in probs.items()}

        for loc, p in probs.items():
            state.set_value(f"confidence/{obj}/{loc}", min(max(float(p), 0.0), 1.0))

    def _update_other_objects(self, state, target: str, detections: List[dict]) -> None:
        searched = state.get_value("target_location")
        if searched is None:
            return
        held = state.get_value("holding_object")
        seen = sorted({obj for obj in StretchState.get_objects() if obj not in (target, held)
                       for det in detections if self._matches(obj, det)})
        locations = StretchState.get_locations()
        for obj in seen:
            for loc in locations:
                state.set_value(f"confidence/{obj}/{loc}", float(loc == searched))
        if seen:
            self._info(f"Also saw {seen} at '{searched}'.")

    def input_variables(self):
        confs = [f"confidence/{obj}/{loc}" for obj in StretchState.get_objects() for loc in StretchState.get_locations()]
        return confs + ["target_object", "target_location", "holding_object"]

    def output_variables(self):
        confs = [f"confidence/{obj}/{loc}" for obj in StretchState.get_objects() for loc in StretchState.get_locations()]
        return confs + ["scan/objects", "target_detection", "target_found"]

    def semantic_description(self):
        return ("Scans the surface in front of the robot looking for the target object, and updates where the "
                "target object and any other objects it sees are believed to be")


class GraspObjectNode(RosServiceNode):
    '''
    Grasps "target_detection". Succeeds when the grasp server confirms the object is held,
    and writes the result to "holding_object" and "gripper_occupied".

    The servers check the grasp themselves before they answer (the grasp verifier or the
    detector looks for the expected label in the gripper, and the pick location is checked
    to be clear), so their answer is the confirmation: success means held ("Confirmed",
    "Held", "Likely held"); "Wrong object: ..." means something else is in the gripper,
    so "gripper_occupied" stays True and the put-back skills set it down where it was
    picked (at the start of the next pass of main_demo_loop); any other failure means empty.
    The finger joint is not used: these grippers' fingers flex around an object, so the
    joint can read nearly closed (an aperture of a few mm) while holding one.

    Objects the detector marks grasp_direction "top" go to the topdown_grasp service, all
    others to grasp_carry (front grasp). Both lift the object and check it in the gripper
    before they return. The request carries the detector's label, which the servers look
    for in the gripper and no longer at the pick location, and its 3D box, which keeps the
    grasp to this object's points and aligns the fingers with its long axis.
    Switches the driver to position mode first.

    Before grasping it records where ("pick/..."): the location, the object meant, the
    point in the map frame (from TF, so it holds wherever the robot drives) and in
    base_link, and the detection's size, so a wrongly picked object can be put back
    there (NavigateToPickLocationNode, MoveOverPickPointNode, PlaceObjectNode).
    '''
    SRV_TYPE = GraspPoint
    DRIVER_MODE = "position"

    WRONG_OBJECT_PREFIX = "Wrong object"     # how the servers' grasp check reports another object held

    def __init__(self, name: str = "GraspObjectNode", service_name: str = "/grasp_carry",
                 topdown_service_name: str = "/topdown_grasp", assumed_width_m: float = 0.0,
                 map_frame: str = "map", **kwargs):
        super().__init__(name, service_name, **kwargs)
        self.map_frame            = map_frame
        self.front_service_name   = service_name
        self.topdown_service_name = topdown_service_name
        self.assumed_width_m      = assumed_width_m  # 0 uses the server's default
        self._clients             = {}

    def ros_setup(self) -> None:
        super().ros_setup()
        self._clients = {self.front_service_name: self._client,
                         self.topdown_service_name: self._ros_node.create_client(GraspPoint, self.topdown_service_name)}
        self._tf = _TfLookup.for_node(self._ros_node)

    def execute(self, state):
        # Pick the server before the base class checks it is ready and calls it.
        if self._future is None and self._response is None:
            det = state.get_value("target_detection")
            top = det is not None and det.get("grasp_direction", "").strip().lower() == "top"
            self.service_name = self.topdown_service_name if top else self.front_service_name
            self._client = self._clients[self.service_name]
        return super().execute(state)

    def make_request(self, state):
        det = state.get_value("target_detection")
        if det is None:
            self._error("No target detection to grasp; run a search first.")
            return Status.FAILURE

        request = GraspPoint.Request()
        request.target.header.frame_id    = det["frame_id"]
        request.target.header.stamp.sec   = det["stamp"]["sec"]
        request.target.header.stamp.nanosec = det["stamp"]["nanosec"]
        request.target.point.x, request.target.point.y, request.target.point.z = det["centroid"]
        request.assumed_width_m = float(self.assumed_width_m)
        # The detector's own label, since that is what the servers detect in the gripper.
        request.expected_label = det["detector_class_name"] or det["class_name"]
        # Detections saved before the box was recorded leave it empty; the servers then use the point alone.
        if "bbox_center" in det:
            box = request.bbox
            box.center.position.x, box.center.position.y, box.center.position.z = det["bbox_center"]
            (box.center.orientation.x, box.center.orientation.y,
             box.center.orientation.z, box.center.orientation.w) = det["bbox_orientation"]
            box.size.x, box.size.y, box.size.z = det["size"]

        self._info(f"Grasping '{det['class_name']}' at {det['centroid']} ({det['frame_id']}) "
                   f"with '{self.service_name}'.")
        self._record_pick(state, det)
        return request

    def _record_pick(self, state, det: dict) -> None:
        '''Where this grasp is aimed, for putting a wrongly picked object back (see the class docstring).'''
        to_map = self._tf.lookup(self.map_frame, det["frame_id"])
        point_map = _TfLookup.apply(to_map, det["centroid"]) if to_map is not None else None
        if point_map is None:
            self._warn(f"No transform from '{det['frame_id']}' to '{self.map_frame}', so the pick point is "
                       f"only kept in '{det['frame_id']}'.")
        state.set_value("pick/location", state.get_value("target_location"))
        state.set_value("pick/object", state.get_value("target_object"))
        state.set_value("pick/point_map", point_map)
        state.set_value("pick/point_base", list(det["centroid"]) if det["frame_id"] == "base_link" else None)
        state.set_value("pick/size", list(det["size"]))

    def on_response(self, state, response):
        if response.success:
            self._info(f"Grasp confirmed by the server: '{response.message}'")
            state.set_value("holding_object", state.get_value("target_object"))
            state.set_value("gripper_occupied", True)
            return Status.SUCCESS
        wrong_object = response.message.startswith(self.WRONG_OBJECT_PREFIX)
        self._warn(f"Grasp failed: '{response.message}'" + (" (something else is in the gripper)" if wrong_object else ""))
        state.set_value("holding_object", None)
        state.set_value("gripper_occupied", wrong_object)
        return Status.FAILURE

    def input_variables(self):
        return ["target_detection", "target_object", "target_location"]

    def output_variables(self):
        return ["holding_object", "gripper_occupied", "pick/location", "pick/object", "pick/point_map",
                "pick/point_base", "pick/size"]

    def semantic_description(self):
        return "Grasps the target object, lifts it, and checks that it is actually held in the gripper"


class IsHoldingObjectNode(_SkillLogging, RosConditionNode):
    '''
    Checks from the gripper aperture that something is in the gripper. The gripper must
    have been commanded closed for this to be meaningful, and it is unreliable on the
    robot: the fingers flex around an object, so the joint can read nearly closed while
    holding one. "gripper_occupied", set from the grasp servers' checks, is reliable.
    '''
    def __init__(self, name: str = "IsHoldingObjectNode", joint_states_topic: str = "/stretch/joint_states",
                 min_held_aperture_m: float = 0.01, timeout_s: float = 3.0):
        super().__init__(name)
        self.min_held_aperture_m = min_held_aperture_m
        self.timeout_s           = timeout_s
        self._gripper            = _GripperMonitor(joint_states_topic)
        self._start_s            = None

    def initialise(self) -> None:
        self._gripper.start(self._ros_node)
        self._start_s = time.monotonic()

    def execute(self, state):
        aperture = self._gripper.aperture_m
        if aperture is None:
            if time.monotonic() - self._start_s > self.timeout_s:
                self._error(f"No gripper state on '{self._gripper.topic}'.")
                return Status.FAILURE
            return Status.RUNNING
        return Status.SUCCESS if aperture > self.min_held_aperture_m else Status.FAILURE

    def terminate(self, new_status) -> None:
        self._gripper.stop(self._ros_node)

    def input_variables(self):
        return []

    def output_variables(self):
        return []

    def semantic_description(self):
        return "Checks whether the robot is holding an object in its gripper"


def _wrongly_held(state) -> bool:
    '''
    Whether the gripper holds something the robot did not mean to pick up: the grasp check
    found the wrong object ("gripper_occupied" but no "holding_object"). The put-back
    skills only act then, so they can run at the start of every pass and do nothing when
    there is nothing to put back, nor set down an object the robot did mean to hold.
    '''
    return bool(state.get_value("gripper_occupied")) and state.get_value("holding_object") is None


class NavigateToPickLocationNode(NavigateToPlaceNode):
    '''
    Drives back to "pick/location", where the last grasp picked up the wrong object the
    robot holds. Already there, the goal is the pose the robot is at, so it finishes at
    once. Succeeds at once, doing nothing, unless a wrong object is held (_wrongly_held).
    '''
    def __init__(self, name: str = "NavigateToPickLocationNode", **kwargs):
        super().__init__(name, **kwargs)

    def execute(self, state):
        if not _wrongly_held(state):
            return Status.SUCCESS
        return super().execute(state)

    def _target(self, state):
        return state.get_value("pick/location")

    def input_variables(self) -> List[str]:
        return ["pick/location", "gripper_occupied", "holding_object"]

    def semantic_description(self):
        return "Navigates the robot back to where it picked up the object it is holding"


class MoveOverPickPointNode(RosActionClientNode):
    '''
    Brings the held object back over the point the last grasp picked it from ("pick/..."),
    for PlaceObjectNode to set it down there, one motion at a time:
      0. unstow the wrist to the grasp servers' carry pose (straight out, level): navigation
         stows it folded over the base, and place_object does not move the wrist,
      1. raise the lift so the object clears the surface (never lowers it),
      2. shift the base along its heading to line the arm up with the point,
      3. extend or retract the arm to reach over it.
    Steps 1-3 are planned once the wrist has settled, from where the gripper then is.
    Succeeds at once, doing nothing, unless a wrong object is held (_wrongly_held).
    The point comes from the map frame, so it holds wherever the robot has driven since;
    without it, from base_link as it was at the grasp, which only holds while the base
    has not moved. Where the point is beyond max_shift_m or the arm's travel, the node
    goes as close as it can and says so; PlaceObjectNode measures the surface under the
    gripper and does not place without one. The gripper's position comes from TF
    (base_link -> link_grasp_center) and the joints from /stretch/joint_states.
    Switches the driver to position mode first.
    '''
    ACTION_TYPE = FollowJointTrajectory
    DRIVER_MODE = "position"
    WRIST_CARRY_POSE = {"joint_wrist_yaw": 0.0, "joint_wrist_pitch": 0.0, "joint_wrist_roll": 0.0}
    WRIST_SETTLE_S = 0.5        # for TF to show the gripper where the wrist put it

    def __init__(self, name: str = "MoveOverPickPointNode",
                 action_name: str = "/stretch_controller/follow_joint_trajectory",
                 joint_states_topic: str = "/stretch/joint_states", map_frame: str = "map",
                 base_frame: str = "base_link", gripper_frame: str = "link_grasp_center",
                 clearance_m: float = 0.05, max_shift_m: float = 0.3,
                 extension_limits_m: tuple = (0.0, 0.5), lift_limits_m: tuple = (0.2, 1.09),
                 move_s: float = 3.0, tolerance_m: float = 0.01, plan_timeout_s: float = 5.0, **kwargs):
        super().__init__(name, action_name, **kwargs)
        self.map_frame          = map_frame
        self.base_frame         = base_frame
        self.gripper_frame      = gripper_frame
        self.clearance_m        = clearance_m          # object bottom above the surface while moving over it
        self.max_shift_m        = max_shift_m
        self.extension_limits_m = extension_limits_m
        self.lift_limits_m      = lift_limits_m
        self.move_s             = move_s
        self.tolerance_m        = tolerance_m
        self.plan_timeout_s     = plan_timeout_s
        self._joints            = _GripperMonitor(joint_states_topic)
        self._tf                = None
        self._steps             = None     # [(joint, position, description)], planned once per run
        self._step              = 0
        self._plan_start_s      = None

    def ros_setup(self) -> None:
        super().ros_setup()
        self._tf = _TfLookup.for_node(self._ros_node)

    def initialise(self) -> None:
        super().initialise()
        self._steps, self._step, self._plan_start_s = None, 0, time.monotonic()
        self._wrist_ready = False

    def execute(self, state):
        if not _wrongly_held(state):
            return Status.SUCCESS
        return super().execute(state)

    def _goal(self, positions: Dict[str, float], description: str):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(positions)
        point = JointTrajectoryPoint(positions=[float(v) for v in positions.values()])
        point.time_from_start.sec = int(self.move_s)
        point.time_from_start.nanosec = int((self.move_s % 1.0) * 1e9)
        goal.trajectory.points = [point]
        self._info(description)
        return goal

    def make_goal(self, state):
        if not self._wrist_ready:
            return self._goal(self.WRIST_CARRY_POSE, "Unstow the wrist: straight out and level.")
        if time.monotonic() - self._plan_start_s < self.WRIST_SETTLE_S:
            return None
        if self._steps is None:
            plan = self._plan(state)
            if plan is None or isinstance(plan, Status):
                return plan
            # The joint states were only needed for the plan. Left subscribed at 20 Hz, they
            # would crowd out the goal results: the runner takes one callback per tick.
            self._joints.stop(self._ros_node)
            if not plan:
                self._info("The gripper is already over the pick point.")
                return Status.SUCCESS
            self._steps = plan
        joint, position, description = self._steps[self._step]
        return self._goal({joint: position}, description)

    def on_result(self, state, goal_status, result):
        joint = self._steps[self._step][0] if self._wrist_ready else "the wrist"
        error = getattr(result, "error_code", FollowJointTrajectory.Result.SUCCESSFUL) if result is not None else 0
        if goal_status != GoalStatus.STATUS_SUCCEEDED or error != FollowJointTrajectory.Result.SUCCESSFUL:
            self._warn(f"Moving '{joint}' failed (goal status {goal_status}, error {error}: "
                       f"'{getattr(result, 'error_string', '')}').")
            return Status.FAILURE
        if not self._wrist_ready:
            # Now plan from where the unstowed gripper is: fresh joint states, settled TF.
            self._reset()
            self._wrist_ready, self._plan_start_s = True, time.monotonic()
            self._joints.start(self._ros_node)
            return Status.RUNNING
        self._step += 1
        if self._step < len(self._steps):
            steps, step = self._steps, self._step
            self._reset()                   # ready to send the next goal on the next tick
            self._steps, self._step = steps, step
            return Status.RUNNING
        return Status.SUCCESS

    def _pick_point(self, state):
        '''The pick point in base_link now, or None to wait for TF, or a Status.'''
        point_map = state.get_value("pick/point_map")
        if point_map is not None:
            to_base = self._tf.lookup(self.base_frame, self.map_frame)
            if to_base is not None:
                return _TfLookup.apply(to_base, point_map)
            if time.monotonic() - self._plan_start_s < self.plan_timeout_s:
                return None
        point_base = state.get_value("pick/point_base")
        if point_base is not None:
            self._warn(f"No pick point in '{self.map_frame}', so using it in '{self.base_frame}' as it was at "
                       f"the grasp; that is only right if the base has not moved since.")
            return list(point_base)
        self._error("No pick point recorded, so not knowing where to put the object back.")
        return Status.FAILURE

    def _plan(self, state):
        point = self._pick_point(state)
        if point is None or isinstance(point, Status):
            return point
        gripper = self._tf.lookup(self.base_frame, self.gripper_frame)
        lift, extension = self._joints.positions.get("joint_lift"), self._joints.positions.get("wrist_extension")
        if gripper is None or lift is None or extension is None:
            if time.monotonic() - self._plan_start_s < self.plan_timeout_s:
                return None
            self._error(f"No {'gripper transform' if gripper is None else 'joint states'} after "
                        f"{self.plan_timeout_s:.0f} s, so unable to plan the move.")
            return Status.FAILURE
        gx, gy, gz = _TfLookup.apply(gripper, [0.0, 0.0, 0.0])
        px, py, pz = point
        steps = []

        # 1. The grasp held the object about its pick point, so with the gripper this high
        # its bottom clears the surface by clearance_m. Only ever raise: lowering could
        # sweep it into the surface edge.
        want_z = pz + self.clearance_m
        if want_z > gz + self.tolerance_m:
            target = min(max(lift + (want_z - gz), self.lift_limits_m[0]), self.lift_limits_m[1])
            steps.append(("joint_lift", target, f"Raise the lift to {target:.2f} m so the object clears the surface."))

        # 2. Line the arm up with the point along the base's heading.
        dx = px - gx
        if abs(dx) > self.tolerance_m:
            shift = min(max(dx, -self.max_shift_m), self.max_shift_m)
            if abs(shift - dx) > self.tolerance_m:
                self._warn(f"The pick point is {dx:+.2f} m along the base; shifting only {shift:+.2f} m, "
                           f"so placing as close to it as possible.")
            steps.append(("translate_mobile_base", shift, f"Shift the base {shift:+.2f} m to line up with the pick point."))

        # 3. Reach over it: the arm extends along the base's -y.
        want = extension + (gy - py)
        target = min(max(want, self.extension_limits_m[0]), self.extension_limits_m[1])
        if abs(target - want) > self.tolerance_m:
            self._warn(f"The pick point needs arm extension {want:.2f} m, beyond "
                       f"{self.extension_limits_m[0]:.2f}-{self.extension_limits_m[1]:.2f} m; "
                       f"going to {target:.2f} m, as close as possible.")
        if abs(target - extension) > self.tolerance_m:
            steps.append(("wrist_extension", target, f"Extend the arm to {target:.2f} m over the pick point."))
        return steps

    def terminate(self, new_status) -> None:
        self._joints.stop(self._ros_node)
        super().terminate(new_status)

    def input_variables(self):
        return ["pick/point_map", "pick/point_base", "gripper_occupied", "holding_object"]

    def output_variables(self):
        return []

    def semantic_description(self):
        return "Moves the held object back over the point it was picked up from"


class PlaceObjectNode(RosServiceNode):
    '''
    Runs the place_object service: measures the surface under the gripper, lowers the
    held object until the surface takes its weight, releases it and lifts clear. It does
    not move the base, arm extension or wrist, so the gripper must be over the place
    already (MoveOverPickPointNode). On success the gripper is empty, and the object, if
    known, is believed to be at "pick/location".
    With only_wrong_object (the default), it succeeds at once, doing nothing, unless a
    wrong object is held (_wrongly_held), so it never sets down the object the robot means
    to fetch. Switches the driver to position mode first.
    '''
    SRV_TYPE = Trigger
    DRIVER_MODE = "position"

    def __init__(self, name: str = "PlaceObjectNode", service_name: str = "/place_object",
                 only_wrong_object: bool = True, **kwargs):
        super().__init__(name, service_name, **kwargs)
        self.only_wrong_object = only_wrong_object

    def execute(self, state):
        if self.only_wrong_object and not _wrongly_held(state):
            return Status.SUCCESS
        return super().execute(state)

    def make_request(self, state):
        self._info("Placing the held object.")
        return Trigger.Request()

    def on_response(self, state, response):
        if not response.success:
            self._warn(f"Place failed: '{response.message}'")
            return Status.FAILURE
        self._info(f"Placed: '{response.message}'")
        obj, location = state.get_value("holding_object"), state.get_value("pick/location")
        if obj is not None and location in StretchState.get_locations():
            for loc in StretchState.get_locations():
                state.set_value(f"confidence/{obj}/{loc}", 1.0 if loc == location else 0.0)
        state.set_value("holding_object", None)
        state.set_value("gripper_occupied", False)
        return Status.SUCCESS

    def input_variables(self):
        return ["holding_object", "gripper_occupied", "pick/location"]

    def output_variables(self):
        confs = [f"confidence/{obj}/{loc}" for obj in StretchState.get_objects() for loc in StretchState.get_locations()]
        return confs + ["holding_object", "gripper_occupied"]

    def semantic_description(self):
        return "Sets the held object down on the surface under the gripper and lets go of it"


class HandoverNode(RosServiceNode):
    '''
    Runs the handover service: presents the held object, waits for a hand on it, releases
    and stows. On success the object's location becomes "Human". If no hand arrives the
    server retracts the arm, still holding the object, and this node fails.

    The handover server must be started with run_once:=false to offer the service.
    Switches the driver to position mode first.
    '''
    SRV_TYPE = Trigger
    DRIVER_MODE = "position"

    def __init__(self, name: str = "HandoverNode", service_name: str = "/handover", **kwargs):
        super().__init__(name, service_name, **kwargs)

    def make_request(self, state):
        self._info("Handing the object over.")
        return Trigger.Request()

    def on_response(self, state, response):
        if not response.success:
            self._warn(f"Handover failed: '{response.message}'")
            return Status.FAILURE

        self._info(f"Handover done: '{response.message}'")
        obj = state.get_value("holding_object") or state.get_value("target_object")
        if obj is not None:
            for loc in StretchState.get_locations():
                state.set_value(f"confidence/{obj}/{loc}", 1.0 if loc == "Human" else 0.0)
        state.set_value("holding_object", None)
        state.set_value("gripper_occupied", False)
        return Status.SUCCESS

    def input_variables(self):
        return ["holding_object", "target_object"]

    def output_variables(self):
        confs = [f"confidence/{obj}/{loc}" for obj in StretchState.get_objects() for loc in StretchState.get_locations()]
        return confs + ["holding_object", "gripper_occupied"]

    def semantic_description(self):
        return "Hands the held object to the human, releasing it once the human's hand is on it"

'''
TEST
'''
class TestNode(ActionNode):
    def __init__(self, name: str = "TestNode"):
        super().__init__(name)

        self.board.register_key("environment", access=py_trees.common.Access.READ)

    def execute(self, state: StretchState):
        state.set_value("recipe/name","caprese_salad")

        return Status.SUCCESS

    def input_variables(self):
        return []

    def output_variables(self):
        return []
