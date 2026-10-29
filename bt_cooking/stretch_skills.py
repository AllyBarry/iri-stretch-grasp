import json
import math
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import py_trees
from ament_index_python.packages import get_package_share_directory
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, ReliabilityPolicy

from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState
from std_srvs.srv import Trigger

from stretch_floor_perception.srv import DetectFloorObjects3D, GraspPoint

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


class RosServiceNode(_SkillLogging, RosActionNode):
    '''
    Base class for a skill that is a single ROS service call.

    Subclasses set SRV_TYPE and implement make_request() and on_response(). A subclass
    that sets DRIVER_MODE switches the driver to that mode before calling the service.
    Service calls cannot be cancelled, so a preempted or timed-out call keeps
    running on the server; its response is discarded.
    '''
    SRV_TYPE = None
    DRIVER_MODE: Optional[str] = None

    def __init__(self, name: str, service_name: str, server_timeout_s: float = 5.0,
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
    '''
    ACTION_TYPE = None
    DRIVER_MODE: Optional[str] = None

    def __init__(self, name: str, action_name: str, server_timeout_s: float = 5.0,
                 result_timeout_s: Optional[float] = None):
        super().__init__(name)
        self.action_name      = action_name
        self.server_timeout_s = server_timeout_s
        self.result_timeout_s = result_timeout_s
        self._client          = None
        self._mode_switch     = _DriverModeSwitch(self.DRIVER_MODE, server_timeout_s) if self.DRIVER_MODE else None
        self._reset()

    def _reset(self) -> None:
        self._start_s       = time.monotonic()
        self._sent_s        = None
        self._goal_future   = None
        self._goal_handle   = None
        self._result_future = None
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
            self._goal_handle   = goal_handle
            self._result_future = goal_handle.get_result_async()
            return Status.RUNNING

        if not self._result_future.done():
            return Status.RUNNING

        wrapped = self._result_future.result()
        return self.on_result(state, wrapped.status, wrapped.result)

    def _timed_out(self) -> bool:
        return self.result_timeout_s is not None and time.monotonic() - self._sent_s > self.result_timeout_s

    def terminate(self, new_status) -> None:
        if self._result_future is not None and not self._result_future.done():
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

    def start(self, ros_node) -> None:
        self.aperture_m = None
        if self._sub is None:
            self._sub = ros_node.create_subscription(JointState, self.topic, self._on_joint_states, 1)

    def stop(self, ros_node) -> None:
        if self._sub is not None:
            ros_node.destroy_subscription(self._sub)
            self._sub = None

    def _on_joint_states(self, msg: JointState) -> None:
        if self.FINGER_JOINT in msg.name:
            finger_rad = msg.position[msg.name.index(self.FINGER_JOINT)]
            self.aperture_m = 2.0 * finger_rad * self.FINGER_LENGTH_M


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
    Switches the driver to navigation mode first.
    '''
    ACTION_TYPE = NavigateToPose
    DRIVER_MODE = "navigation"

    def __init__(self, name: str = "NavigateToPlaceNode", location: Optional[str] = None,
                 location_poses: Optional[Dict[str, Dict[str, float]]] = None,
                 action_name: str = "/navigate_to_pose", map_frame: str = "map", **kwargs):
        super().__init__(name, action_name, **kwargs)
        self.location       = location
        self.location_poses = location_poses
        self.map_frame      = map_frame

    def ros_setup(self) -> None:
        super().ros_setup()
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
        return [] if self.location is not None else ["target_location"]

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
    Grabs the latest camera images, sends them to the floor_object_detector service and
    writes every detected object to "scan/objects" as a list of plain dicts (the state is
    JSON-logged every tick, so it cannot hold ROS messages).

    Images are only taken once settle_s has passed since the node started, so the camera
    has caught up with any preceding head or base motion. Centroids are in the detector's
    base_frame (base_link), so they are only valid until the base moves.
    '''
    SRV_TYPE = DetectFloorObjects3D

    def __init__(self, name: str = "ScanSurfaceNode", service_name: str = "/detect_floor_objects_3d",
                 color_topic: str = "/camera/color/image_raw/compressed",
                 depth_topic: str = "/camera/aligned_depth_to_color/image_raw/compressedDepth",
                 camera_info_topic: str = "/camera/color/camera_info",
                 settle_s: float = 0.5, image_timeout_s: float = 5.0, min_score: float = 0.0, **kwargs):
        super().__init__(name, service_name, **kwargs)
        self.topics = {"color": (CompressedImage, color_topic),
                       "depth": (CompressedImage, depth_topic),
                       "info":  (CameraInfo, camera_info_topic)}
        self.settle_s        = settle_s
        self.image_timeout_s = image_timeout_s
        self.min_score       = min_score
        self._subs           = []
        self._latest         = {}

    def initialise(self) -> None:
        super().initialise()
        self._start_camera()

    def _start_camera(self) -> None:
        # Subscribed only while scanning: the runner handles one callback per tick,
        # so permanent image subscriptions would crowd out other callbacks.
        self._stop_camera()
        self._latest = {}
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        for key, (msg_type, topic) in self.topics.items():
            self._subs.append(self._ros_node.create_subscription(
                msg_type, topic, lambda msg, key=key: self._on_image(key, msg), qos))

    def _stop_camera(self) -> None:
        for sub in self._subs:
            self._ros_node.destroy_subscription(sub)
        self._subs = []

    def _on_image(self, key: str, msg) -> None:
        if time.monotonic() - self._start_s >= self.settle_s:
            self._latest[key] = msg

    def make_request(self, state):
        if len(self._latest) < len(self.topics):
            if time.monotonic() - self._start_s > self.settle_s + self.image_timeout_s:
                missing = [self.topics[key][1] for key in self.topics if key not in self._latest]
                self._error(f"No camera messages received on {missing}.")
                return Status.FAILURE
            return None

        self._stop_camera()
        request = DetectFloorObjects3D.Request()
        request.color_msg = self._latest["color"]
        request.depth_msg = self._latest["depth"]
        request.info_msg  = self._latest["info"]
        self._info("Scanning the surface.")
        return request

    def on_response(self, state, response):
        detections = [self._to_dict(det) for det in response.detections.detections if det.score >= self.min_score]
        state.set_value("scan/objects", detections)
        self._info(f"Detected {len(detections)} object(s): {[det['class_name'] for det in detections]}")
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

    def terminate(self, new_status) -> None:
        self._stop_camera()
        super().terminate(new_status)

    def input_variables(self):
        return []

    def output_variables(self):
        return ["scan/objects"]

    def semantic_description(self):
        return "Scans the surface in front of the robot with its camera and lists the objects it can see"


class SearchForObjectNode(ScanSurfaceNode):
    '''
    Scans the surface and looks for "target_object" among the detections. On success the
    best match is written to "target_detection" for GraspObjectNode.

    Also updates the belief about where the target is: seeing it at "target_location"
    makes that location certain; missing it there scales that location's probability by
    (1 - detection_prob) and renormalises, since the detector can miss objects.

    Every other known object seen in the scan is also now believed to be at
    "target_location". Objects that are not seen are left as they were: a single scan
    only covers part of a surface. The object in the gripper is never moved.

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
    Grasps "target_detection", then confirms the grasp from the gripper aperture. Succeeds
    only if the object is confirmed in the gripper, and writes the result to "holding_object".

    Objects the detector marks grasp_direction "top" go to the topdown_grasp service, all
    others to grasp_carry (front grasp). Both lift the object and check it in the gripper
    before they return. The request carries the detector's label, which the servers look
    for in the gripper and no longer at the pick location, and its 3D box, which keeps the
    grasp to this object's points and aligns the fingers with its long axis.
    Switches the driver to position mode first.
    '''
    SRV_TYPE = GraspPoint
    DRIVER_MODE = "position"

    def __init__(self, name: str = "GraspObjectNode", service_name: str = "/grasp_carry",
                 topdown_service_name: str = "/topdown_grasp",
                 joint_states_topic: str = "/stretch/joint_states", min_held_aperture_m: float = 0.01,
                 confirm_timeout_s: float = 3.0, assumed_width_m: float = 0.0, **kwargs):
        super().__init__(name, service_name, **kwargs)
        self.front_service_name   = service_name
        self.topdown_service_name = topdown_service_name
        self.min_held_aperture_m  = min_held_aperture_m
        self.confirm_timeout_s    = confirm_timeout_s
        self.assumed_width_m      = assumed_width_m  # 0 uses the server's default
        self._gripper             = _GripperMonitor(joint_states_topic)
        self._confirm_start_s     = None
        self._clients             = {}

    def ros_setup(self) -> None:
        super().ros_setup()
        self._clients = {self.front_service_name: self._client,
                         self.topdown_service_name: self._ros_node.create_client(GraspPoint, self.topdown_service_name)}

    def initialise(self) -> None:
        super().initialise()
        self._confirm_start_s = None

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
        return request

    def on_response(self, state, response):
        if self._confirm_start_s is None:
            if not response.success:
                self._warn(f"Grasp failed: '{response.message}'")
                state.set_value("holding_object", None)
                return Status.FAILURE
            # Subscribe only now, so the aperture reading is from after the grasp.
            self._gripper.start(self._ros_node)
            self._confirm_start_s = time.monotonic()

        aperture = self._gripper.aperture_m
        if aperture is None:
            if time.monotonic() - self._confirm_start_s > self.confirm_timeout_s:
                self._error(f"No gripper state on '{self._gripper.topic}', so unable to confirm the grasp.")
                state.set_value("holding_object", None)
                return Status.FAILURE
            return Status.RUNNING

        self._gripper.stop(self._ros_node)
        held = aperture > self.min_held_aperture_m
        state.set_value("holding_object", state.get_value("target_object") if held else None)
        if held:
            self._info(f"Grasp confirmed: gripper aperture {aperture:.3f} m.")
            return Status.SUCCESS
        self._warn(f"Grasp not confirmed: gripper closed to {aperture:.3f} m (needs > {self.min_held_aperture_m:.3f} m).")
        return Status.FAILURE

    def terminate(self, new_status) -> None:
        self._gripper.stop(self._ros_node)
        self._confirm_start_s = None
        super().terminate(new_status)

    def input_variables(self):
        return ["target_detection", "target_object"]

    def output_variables(self):
        return ["holding_object"]

    def semantic_description(self):
        return "Grasps the target object, lifts it, and checks that it is actually held in the gripper"


class IsHoldingObjectNode(_SkillLogging, RosConditionNode):
    '''
    Checks from the gripper aperture that something is in the gripper, e.g. before a
    handover. The gripper must have been commanded closed for this to be meaningful.
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
        return Status.SUCCESS

    def input_variables(self):
        return ["holding_object", "target_object"]

    def output_variables(self):
        confs = [f"confidence/{obj}/{loc}" for obj in StretchState.get_objects() for loc in StretchState.get_locations()]
        return confs + ["holding_object"]

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
