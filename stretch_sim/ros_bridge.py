'''
ROS 2 interface of the simulator. It offers the same services, actions and topics as
the real Stretch stack, backed by the simulator, so the cooking behaviour tree runs
against it unchanged:

    /switch_to_{position,navigation,trajectory}_mode  std_srvs/Trigger            (stretch_driver)
    /navigate_to_pose                                  nav2_msgs/NavigateToPose     (Nav2)
    /stretch_controller/follow_joint_trajectory        control_msgs/FollowJointTrajectory: head, lift,
                                                       wrist_extension, translate_mobile_base
    /place_object                                      std_srvs/Trigger             (place_object)
    /tf                                                map -> odom -> base_link -> link_grasp_center
    /detect_floor_objects_3d                           DetectFloorObjects3D         (floor_object_detector)
    /scan_surface                                      ScanSurface                  (scan_surface)
    /grasp_carry, /topdown_grasp                       GraspPoint                   (grasp_carry, topdown_grasp)
    /handover                                          std_srvs/Trigger             (handover)
    /stretch/joint_states                              sensor_msgs/JointState       published
    /camera/color/image_raw/compressed,                placeholder images, published so
    /camera/aligned_depth_to_color/image_raw/compressedDepth, /camera/color/camera_info
                                                       the scan node has something to send

An interface whose message package is not installed is skipped with a warning, and
listed in `missing`. With serve_robot=False (robot mode, see robot_link.py) none of them
are offered, since the real robot offers them; the bridge then only follows the BT's state.

Requests block until the simulated operation finishes, on a multi-threaded executor,
so the simulation must be stepped by something else (SimController).
'''
import importlib
import json
import math
import threading
import time
from typing import Callable, List, Optional

import rclpy
import rclpy.signals
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState
from action_msgs.srv import CancelGoal
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger

from stretch_sim.events import INFO
from stretch_sim.geometry import Pose2D
from stretch_sim.operations import OpStatus
from stretch_sim.robot import DRIVER_MODES
from stretch_sim.simulator import KinematicSimulator

HEAD_JOINTS = ("joint_head_pan", "joint_head_tilt")
ARM_JOINTS = ("joint_lift", "wrist_extension", "translate_mobile_base")
WRIST_JOINTS = ("joint_wrist_yaw", "joint_wrist_pitch", "joint_wrist_roll")   # not modelled: logged only
GRIPPER_JOINT = "joint_gripper_finger_left"
# Everything the sim may offer: what the robot offers with bringup_stretch.sh running. The
# sim must not offer anything else, or a BT that works against it could fail on the robot.
ROBOT_INTERFACES = frozenset({
    "/switch_to_position_mode", "/switch_to_navigation_mode", "/switch_to_trajectory_mode",
    "/switch_to_gamepad_mode",                                      # stretch_driver
    "/navigate_to_pose",                                            # Nav2
    "/stretch_controller/follow_joint_trajectory",                  # stretch_driver
    "/detect_floor_objects_3d",                                     # floor_object_detector
    "/scan_surface",                                                # scan_surface
    "/grasp_carry", "/topdown_grasp",                               # grasp servers
    "/handover",                                                    # handover (run_once:=false)
    "/place_object",                                                # place_object
    "/tf",                                                          # amcl, stretch_driver, robot_state_publisher
    "/stretch/joint_states",                                        # stretch_driver
    "/camera/color/image_raw/compressed", "/camera/aligned_depth_to_color/image_raw/compressedDepth",
    "/camera/color/camera_info",                                    # realsense
})
# The real robot's actions, whose goals robot mode cancels when the viewer stops the robot.
ROBOT_ACTIONS = ("/navigate_to_pose", "/stretch_controller/follow_joint_trajectory")
BT_STATE_TOPIC = "/cooking_bt/state"     # published by bt_cooking's StretchEnvironment
BT_RESET_SERVICE = "/cooking_bt/reset_memory"   # offered by bt_cooking's StretchEnvironment


def _optional(module: str, name: str):
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        return None


def yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _duration_s(d) -> float:
    return float(d.sec) + float(d.nanosec) * 1e-9


class SimRosBridge(Node):
    def __init__(self, sim: KinematicSimulator, node_name: str = "stretch_sim", poll_s: float = 0.02,
                 camera_hz: float = 5.0, joint_states_hz: float = 20.0, serve_robot: bool = True,
                 on_bt_state: Optional[Callable[[dict], None]] = None, context=None):
        super().__init__(node_name, context=context)
        self.sim = sim
        self.poll_s = poll_s
        self.group = ReentrantCallbackGroup()
        self.offered: List[str] = []
        self.missing: List[str] = []
        self.on_bt_state = on_bt_state     # called with each state the BT publishes, after the sim records it

        if serve_robot:
            self._serve_robot(camera_hz, joint_states_hz)
        else:
            self._follow_robot()

        # Not part of the robot: what a behaviour tree run against the sim believes, for the viewer.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(String, BT_STATE_TOPIC, self._on_bt_state, latched, callback_group=self.group)
        self._bt_reset = self.create_client(Trigger, BT_RESET_SERVICE, callback_group=self.group)

        for item in self.missing:
            self.get_logger().warn(f"Not offered: {item}")
        if serve_robot:
            sim.emit(INFO, "ros", f"ROS interface up: {len(self.offered)} offered, {len(self.missing)} missing",
                     offered=self.offered, missing=self.missing)
        else:
            sim.emit(INFO, "ros", f"following the behaviour tree on {BT_STATE_TOPIC}; the real robot offers its interfaces")

    def _serve_robot(self, camera_hz: float, joint_states_hz: float) -> None:
        for mode in DRIVER_MODES:
            self._service(Trigger, f"/switch_to_{mode}_mode", lambda req, res, mode=mode: self._switch_mode(mode, res))
        self._service(Trigger, "/handover", self._handover)
        self._service(Trigger, "/place_object", self._place)

        NavigateToPose = _optional("nav2_msgs.action", "NavigateToPose")
        FollowJointTrajectory = _optional("control_msgs.action", "FollowJointTrajectory")
        DetectFloorObjects3D = _optional("stretch_floor_perception.srv", "DetectFloorObjects3D")
        GraspPoint = _optional("stretch_floor_perception.srv", "GraspPoint")
        ScanSurface = _optional("stretch_floor_perception.srv", "ScanSurface")
        self._DetectedObject3D = _optional("stretch_floor_perception.msg", "DetectedObject3D")
        self._NavigateToPose, self._FollowJointTrajectory = NavigateToPose, FollowJointTrajectory

        if NavigateToPose:
            self._action(NavigateToPose, "/navigate_to_pose", self._navigate)
        else:
            self.missing.append("/navigate_to_pose (nav2_msgs not installed)")
        if FollowJointTrajectory:
            self._action(FollowJointTrajectory, "/stretch_controller/follow_joint_trajectory", self._follow_trajectory)
        else:
            self.missing.append("/stretch_controller/follow_joint_trajectory (control_msgs not installed)")
        if DetectFloorObjects3D and self._DetectedObject3D:
            self._service(DetectFloorObjects3D, "/detect_floor_objects_3d", self._detect)
        else:
            self.missing.append("/detect_floor_objects_3d (stretch_floor_perception interfaces not built)")
        if ScanSurface and self._DetectedObject3D:
            self._service(ScanSurface, "/scan_surface", self._scan_surface)
        else:
            self.missing.append("/scan_surface (stretch_floor_perception interfaces not built)")
        if GraspPoint:
            self._service(GraspPoint, "/grasp_carry", lambda req, res: self._grasp(req, res, "front"))
            self._service(GraspPoint, "/topdown_grasp", lambda req, res: self._grasp(req, res, "top"))
        else:
            self.missing.append("/grasp_carry, /topdown_grasp (stretch_floor_perception interfaces not built)")

        self.joint_pub = self._publisher(JointState, "/stretch/joint_states", 10)
        self.color_pub = self._publisher(CompressedImage, "/camera/color/image_raw/compressed", 1)
        self.depth_pub = self._publisher(CompressedImage, "/camera/aligned_depth_to_color/image_raw/compressedDepth", 1)
        self.info_pub = self._publisher(CameraInfo, "/camera/color/camera_info", 1)
        self.offered += ["/stretch/joint_states", "/camera/... (placeholder images)"]
        self.create_timer(1.0 / joint_states_hz, self._publish_joint_states, callback_group=self.group)
        self.create_timer(1.0 / camera_hz, self._publish_camera, callback_group=self.group)
        self._check_offered_by_robot("/tf")
        from tf2_ros import TransformBroadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        self.offered.append("/tf")
        self.create_timer(1.0 / joint_states_hz, self._publish_tf, callback_group=self.group)

    def _check_offered_by_robot(self, name: str) -> None:
        if name not in ROBOT_INTERFACES:
            raise ValueError(f"the sim must not offer {name}: the robot does not (see ROBOT_INTERFACES)")

    def _publisher(self, msg_type, name, depth):
        self._check_offered_by_robot(name)
        return self.create_publisher(msg_type, name, depth)

    def _service(self, srv_type, name, callback):
        self._check_offered_by_robot(name)
        self.create_service(srv_type, name, callback, callback_group=self.group)
        self.offered.append(name)

    def _action(self, action_type, name, execute):
        self._check_offered_by_robot(name)
        ActionServer(self, action_type, name, execute_callback=execute,
                     goal_callback=lambda goal: GoalResponse.ACCEPT,
                     cancel_callback=lambda goal_handle: CancelResponse.ACCEPT,
                     callback_group=self.group)
        self.offered.append(name)

    def _wait(self, op, goal_handle=None, on_poll=None) -> bool:
        '''Waits for a simulated operation; returns False if the ROS client canceled it.'''
        while not op.done:
            if goal_handle is not None and goal_handle.is_cancel_requested:
                self.sim.cancel(op)
                return False
            if on_poll:
                on_poll()
            time.sleep(self.poll_s)
        return True

    '''
    DRIVER, HANDOVER
    '''
    def _switch_mode(self, mode: str, res):
        res.success, res.message = self.sim.set_driver_mode(mode)
        return res

    def _place(self, req, res):
        op = self.sim.place()
        self._wait(op)
        res.success = op.status == OpStatus.SUCCEEDED
        res.message = op.message or op.status.value
        return res

    def _handover(self, req, res):
        op = self.sim.handover()
        self._wait(op)
        handed = op.status == OpStatus.SUCCEEDED and bool(op.result.get("handed_over"))
        res.success = handed
        res.message = op.message or op.status.value
        return res

    '''
    NAVIGATION
    '''
    def _navigate(self, goal_handle):
        NavigateToPose = self._NavigateToPose
        pose = goal_handle.request.pose
        if pose.header.frame_id not in ("map", ""):
            self.get_logger().warn(f"Navigation goal in frame '{pose.header.frame_id}'; the sim treats it as 'map'.")
        goal = Pose2D(pose.pose.position.x, pose.pose.position.y, yaw_from_quaternion(pose.pose.orientation))
        op = self.sim.navigate_to(goal)
        start = time.monotonic()

        def feedback():
            fb = NavigateToPose.Feedback()
            r = self.sim.robot.pose
            fb.current_pose.header.frame_id = "map"
            fb.current_pose.pose.position.x, fb.current_pose.pose.position.y = r.x, r.y
            fb.current_pose.pose.orientation.z, fb.current_pose.pose.orientation.w = math.sin(r.yaw / 2), math.cos(r.yaw / 2)
            if hasattr(op, "distance_remaining"):
                fb.distance_remaining = float(op.distance_remaining(r))
            elapsed = time.monotonic() - start
            fb.navigation_time.sec, fb.navigation_time.nanosec = int(elapsed), int((elapsed % 1) * 1e9)
            goal_handle.publish_feedback(fb)

        result = NavigateToPose.Result()
        if not self._wait(op, goal_handle, feedback):
            goal_handle.canceled()
            return result
        if op.status == OpStatus.SUCCEEDED:
            goal_handle.succeed()
        else:
            if hasattr(result, "error_msg"):
                result.error_msg = op.message
            if hasattr(result, "error_code"):
                result.error_code = 9000   # Nav2's "unknown" error code
            goal_handle.abort()
        return result

    '''
    HEAD, LIFT, ARM
    '''
    def _follow_trajectory(self, goal_handle):
        '''Head, lift, arm extension and base translation goals; each joint goes to its last point.'''
        FollowJointTrajectory = self._FollowJointTrajectory
        traj = goal_handle.request.trajectory
        result = FollowJointTrajectory.Result()
        supported = HEAD_JOINTS + ARM_JOINTS + WRIST_JOINTS + (GRIPPER_JOINT,)
        unsupported = [j for j in traj.joint_names if j not in supported]
        if unsupported or not traj.points:
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS if unsupported else FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = (f"the sim only moves {list(supported)}, not {unsupported}" if unsupported
                                   else "trajectory has no points")
            self.sim.emit(INFO, "ros", f"follow_joint_trajectory rejected: {result.error_string}")
            goal_handle.abort()
            return result

        last = traj.points[-1]
        target = dict(zip(traj.joint_names, last.positions))
        ops = []
        if any(j in target for j in HEAD_JOINTS):
            ops.append(self.sim.move_head(target.get("joint_head_pan", self.sim.robot.head_pan),
                                          target.get("joint_head_tilt", self.sim.robot.head_tilt),
                                          max(_duration_s(last.time_from_start), 0.1)))
        if any(j in target for j in ARM_JOINTS + WRIST_JOINTS + (GRIPPER_JOINT,)):
            finger = target.get(GRIPPER_JOINT)
            ops.append(self.sim.move_arm(
                lift=target.get("joint_lift"), extension=target.get("wrist_extension"),
                translate=target.get("translate_mobile_base"),
                aperture=None if finger is None else 2.0 * finger * self.sim.geom.finger_length_m,
                wrist={j: target[j] for j in WRIST_JOINTS if j in target}))
        for op in ops:
            if not self._wait(op, goal_handle):
                for other in ops:
                    self.sim.cancel(other)
                goal_handle.canceled()
                return result
        failed = [op for op in ops if op.status != OpStatus.SUCCEEDED]
        if not failed:
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            goal_handle.succeed()
        else:
            result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
            result.error_string = "; ".join(op.message for op in failed)
            goal_handle.abort()
        return result

    '''
    PERCEPTION, GRASPING
    '''
    def _detection_msg(self, d, stamp):
        m = self._DetectedObject3D()
        m.header.frame_id = "base_link"
        m.header.stamp = stamp
        m.class_name = d.label
        m.detector_class_name = d.label
        m.pick_up = "yes"
        m.grasp_direction = d.grasp_direction
        m.score = float(d.score)
        m.centroid.x, m.centroid.y, m.centroid.z = (float(v) for v in d.centroid)
        m.bbox3d.center.position.x, m.bbox3d.center.position.y, m.bbox3d.center.position.z = (float(v) for v in d.centroid)
        m.bbox3d.center.orientation.w = 1.0
        m.bbox3d.size.x, m.bbox3d.size.y, m.bbox3d.size.z = (float(v) for v in d.size)
        return m

    def _detect(self, req, res):
        stamp = self.get_clock().now().to_msg()
        res.detections.header.frame_id = "base_link"
        res.detections.header.stamp = stamp
        res.detections.detections = [self._detection_msg(d, stamp) for d in self.sim.detect()]
        return res

    def _scan_surface(self, req, res):
        '''
        As the scan_surface node: look at the named surface, detect, and keep only what is
        on it. The real node sweeps the head over several views; the sim's scan covers
        the whole surface, so one head move (one view) stands for the sweep.
        '''
        surfaces = {"_".join(n.lower().split()): n for n, loc in self.sim.world.locations.items() if loc.kind == "surface"}
        name = "_".join(req.surface_name.strip().lower().split())
        location = surfaces.get(name)
        if name and location is None:
            res.success = False
            res.message = f"no surface '{name}' (known: {', '.join(sorted(surfaces)) or 'none'})"
            self.sim.emit(INFO, "ros", f"scan_surface: {res.message}")
            return res

        pan, tilt = self._head_towards(location)
        op = self.sim.move_head(pan, tilt, 1.0)
        self._wait(op)
        if op.status != OpStatus.SUCCEEDED:
            res.success = False
            res.message = f"head goal pan={pan:.2f} tilt={tilt:.2f} failed: {op.message}"
            return res
        detections = self.sim.detect()
        if location is not None:
            rect = self.sim.world.locations[location].rect
            detections = [d for d in detections if rect.contains(*d.world_xy)]
        stamp = self.get_clock().now().to_msg()
        msgs = [self._detection_msg(d, stamp) for d in detections]
        res.views_scanned = 1
        res.detections.header.frame_id = "base_link"
        res.detections.header.stamp = stamp
        res.detections.detections = msgs
        target = req.target_label.strip().lower()
        if target:
            matches = [m for m in msgs if target in m.class_name.lower() or target in m.detector_class_name.lower()]
            if matches:
                res.found = True
                res.target = max(matches, key=lambda m: m.score)
        res.success = True
        names = ", ".join(f"'{m.class_name}'" for m in msgs) or "none"
        where = f" on '{name}'" if name else ""
        if target:
            res.message = (f"found '{res.target.class_name}'{where}" if res.found
                           else f"'{target}' not found{where} in 1 views; saw {names}")
        else:
            res.message = f"{len(msgs)} objects{where} in 1 views: {names}"
        return res

    def _head_towards(self, location):
        '''Head pan and tilt that look at a location (the arm side, if none), within the head's pan range.'''
        r = self.sim.robot.pose
        if location is None:
            return -math.pi / 2.0, -0.7
        c = self.sim.world.locations[location].rect
        pan = math.atan2(c.cy - r.y, c.cx - r.x) - r.yaw
        pan = (pan + math.pi) % (2.0 * math.pi) - math.pi
        if pan > 1.5:       # the pan joint reaches -3.9..1.5 rad
            pan -= 2.0 * math.pi
        return pan, -0.7

    def _grasp(self, req, res, direction: str):
        '''
        Answers as the grasp servers do after their grasp check: success only with an
        object in the gripper, and not when expected_label names a different one. The
        sim grasps at the target point; the box is only logged.
        '''
        frame = req.target.header.frame_id
        if frame not in ("base_link", ""):
            self.get_logger().warn(f"Grasp target in frame '{frame}'; the sim treats it as 'base_link'.")
        if req.target.header.stamp.sec <= 0:
            # grasp_carry and topdown_grasp refuse unstamped targets.
            res.success, res.message = False, "No point cloud available yet, so unable to grasp."
            self.sim.emit(INFO, "ros", f"grasp rejected: {res.message}")
            return res
        size = req.bbox.size
        if size.x > 0.0 and size.y > 0.0:
            self.sim.emit(INFO, "ros", f"grasp request has a box of {size.x:.3f} x {size.y:.3f} x {size.z:.3f} m",
                          bbox_size=[round(size.x, 3), round(size.y, 3), round(size.z, 3)])
        p = req.target.point
        op = self.sim.grasp((p.x, p.y, p.z), direction=direction, expected_label=req.expected_label)
        self._wait(op)
        held = op.result.get("held")
        if op.status != OpStatus.SUCCEEDED:
            res.success, res.message = False, op.message or op.status.value
        elif not held:
            res.success, res.message = False, f"Not confirmed: {op.message}"
        elif op.result.get("wrong_object"):
            res.success, res.message = False, f"Wrong object: {op.message}"
        elif req.expected_label:
            res.success, res.message = True, f"Confirmed: {op.message}"
        else:
            res.success, res.message = True, f"Held (item not identified): {op.message}"
        return res

    def reset_bt_memory(self, timeout_s: float = 5.0):
        '''Asks a running behaviour tree to forget its beliefs. Returns (success, message).'''
        response = self._call(self._bt_reset, Trigger.Request(), timeout_s)
        if isinstance(response, str):
            return False, response.replace("not offered", "offered by no behaviour tree")
        return bool(response.success), response.message

    def _call(self, client, request, timeout_s: float):
        '''Calls a service and waits for it; returns the response, or a string saying why there is none.'''
        if not client.wait_for_service(timeout_sec=min(timeout_s, 2.0)):
            return f"{client.srv_name} not offered"
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda f: done.set())
        if not done.wait(timeout_s):
            client.remove_pending_request(future)
            return f"no answer from {client.srv_name} within {timeout_s:.0f} s"
        return future.result()

    '''
    STOPPING THE REAL ROBOT (robot mode)
    '''
    def _follow_robot(self) -> None:
        '''Clients for stopping the real robot, and its runstop state. Nothing is offered.'''
        self._cancel_clients = [self.create_client(CancelGoal, f"{name}/_action/cancel_goal", callback_group=self.group)
                                for name in ROBOT_ACTIONS]
        self._stop_client = self.create_client(Trigger, "/stop_the_robot", callback_group=self.group)
        self._runstop_client = self.create_client(SetBool, "/runstop", callback_group=self.group)
        self.runstopped: Optional[bool] = None
        self.create_subscription(Bool, "/is_runstopped", self._on_runstopped, 10, callback_group=self.group)

    def _on_runstopped(self, msg) -> None:
        if msg.data != self.runstopped:
            self.runstopped = msg.data
            self.sim.emit(INFO, "robot", "the real robot is runstopped" if msg.data else "the real robot's runstop is released")

    def _call_all(self, calls, timeout_s: float) -> list:
        '''_call for each (client, request) at once, so one missing service does not delay the others.'''
        results = [None] * len(calls)

        def call(i, client, request):
            results[i] = self._call(client, request, timeout_s)
        threads = [threading.Thread(target=call, args=(i, c, r), daemon=True) for i, (c, r) in enumerate(calls)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout_s + 3.0)
        return [r if r is not None else "no answer" for r in results]

    @staticmethod
    def _cancel_summary(client, response) -> str:
        action = client.srv_name.rsplit("/_action/", 1)[0]
        if isinstance(response, str):
            return response
        if response.return_code == CancelGoal.Response.ERROR_NONE:
            return f"{action}: canceled {len(response.goals_canceling)} goal(s)"
        if response.return_code == CancelGoal.Response.ERROR_REJECTED and not response.goals_canceling:
            return f"{action}: no goals"
        return f"{action}: cancel failed (code {response.return_code})"

    def cancel_robot_goals(self, timeout_s: float = 3.0) -> List[str]:
        '''Cancels every goal of the robot's actions (Nav2, trajectories), whoever sent it.'''
        # A zero goal id and time cancels all goals.
        results = self._call_all([(c, CancelGoal.Request()) for c in self._cancel_clients], timeout_s)
        return [self._cancel_summary(c, r) for c, r in zip(self._cancel_clients, results)]

    def stop_robot(self, runstop: bool = False, timeout_s: float = 3.0):
        '''
        Stops the real robot: cancels its action goals and halts the current motion, and
        with runstop also runstops it, which is the only way to stop a grasp, handover or
        scan service part way (they have no cancel). All at once. Returns (success, message).
        '''
        calls = [(c, CancelGoal.Request()) for c in self._cancel_clients]
        calls.append((self._stop_client, Trigger.Request()))
        if runstop:
            calls.insert(0, (self._runstop_client, SetBool.Request(data=True)))    # sent first: it stops everything
        results = self._call_all(calls, timeout_s)
        parts, ok = [], True
        if runstop:
            response = results.pop(0)
            ok &= not isinstance(response, str) and response.success
            parts.append(response if isinstance(response, str) else "runstopped")
        stop = results.pop()
        parts += [self._cancel_summary(c, r) for c, r in zip(self._cancel_clients, results)]
        ok &= not isinstance(stop, str) and stop.success
        parts.append(stop if isinstance(stop, str) else f"/stop_the_robot: {stop.message or 'stopped'}")
        message = "; ".join(parts)
        self.sim.emit(INFO, "robot", f"{'runstopped' if runstop else 'stopped'} the real robot: {message}", ok=ok)
        return ok, message

    def release_runstop(self, timeout_s: float = 3.0):
        response = self._call(self._runstop_client, SetBool.Request(data=False), timeout_s)
        if isinstance(response, str):
            return False, response
        self.sim.emit(INFO, "robot", "released the real robot's runstop")
        return bool(response.success), response.message

    def _on_bt_state(self, msg) -> None:
        try:
            payload = json.loads(msg.data)
        except ValueError:
            self.get_logger().warn(f"Unreadable message on {BT_STATE_TOPIC}")
            return
        self.sim.set_bt_state(payload)
        if self.on_bt_state is not None:
            self.on_bt_state(payload)

    '''
    TOPICS
    '''
    def _publish_joint_states(self):
        with self.sim.lock:
            r = self.sim.robot
            finger = r.finger_joint_rad(self.sim.geom)
            values = {"joint_head_pan": r.head_pan, "joint_head_tilt": r.head_tilt, "joint_lift": r.lift,
                      "wrist_extension": r.arm_extension, "joint_gripper_finger_left": finger,
                      "joint_gripper_finger_right": finger}
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(values)
        msg.position = [float(v) for v in values.values()]
        self.joint_pub.publish(msg)

    def _publish_tf(self):
        '''
        The frames the robot's TF has that the BT uses: map -> odom (identity: the sim
        does not drift), odom -> base_link (the robot's pose) and base_link ->
        link_grasp_center (the gripper: out to the right by the arm's reach, at lift height).
        '''
        from geometry_msgs.msg import TransformStamped
        with self.sim.lock:
            r = self.sim.robot
            pose, reach, lift = r.pose.copy(), self.sim.geom.arm_min_reach + r.arm_extension, r.lift
        stamp = self.get_clock().now().to_msg()
        frames = (("map", "odom", 0.0, 0.0, 0.0, 0.0), ("odom", "base_link", pose.x, pose.y, 0.0, pose.yaw),
                  ("base_link", "link_grasp_center", 0.0, -reach, lift, 0.0))
        msgs = []
        for parent, child, x, y, z, yaw in frames:
            t = TransformStamped()
            t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, parent, child
            t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = x, y, z
            t.transform.rotation.z, t.transform.rotation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
            msgs.append(t)
        self.tf_broadcaster.sendTransform(msgs)

    def _publish_camera(self):
        stamp = self.get_clock().now().to_msg()
        color = CompressedImage(format="jpeg", data=b"\xff\xd8\xff\xd9")
        depth = CompressedImage(format="16UC1; compressedDepth", data=b"\x00")
        info = CameraInfo(width=480, height=640)
        for msg, frame in ((color, "camera_color_optical_frame"), (depth, "camera_color_optical_frame"),
                           (info, "camera_color_optical_frame")):
            msg.header.stamp = stamp
            msg.header.frame_id = frame
        self.color_pub.publish(color)
        self.depth_pub.publish(depth)
        self.info_pub.publish(info)


class RosBridgeRunner:
    '''
    Runs the bridge on a background thread, in its own ROS context, so the sim can stop it
    and start another on a different domain (see ros_link.py).
    '''
    def __init__(self, sim: KinematicSimulator, threads: int = 8, domain_id: Optional[int] = None, **bridge_args):
        self.context = rclpy.Context()
        # rclpy's own handlers would only shut ROS down on Ctrl-C or SIGTERM and leave
        # the rest of the simulator (e.g. the web viewer) running; run_sim handles them.
        rclpy.init(context=self.context, domain_id=domain_id,
                   signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
        self.bridge = SimRosBridge(sim, context=self.context, **bridge_args)
        self.executor = MultiThreadedExecutor(num_threads=threads, context=self.context)
        self.executor.add_node(self.bridge)
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self.executor.spin, name="ros-bridge", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.executor.shutdown()
        if self._thread is not None:
            # Destroying the node while the executor thread still spins aborts the process.
            self._thread.join(timeout=5.0)
        self.bridge.destroy_node()
        self.context.try_shutdown()
