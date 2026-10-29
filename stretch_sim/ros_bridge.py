'''
ROS 2 interface of the simulator. It offers the same services, actions and topics as
the real Stretch stack, backed by the simulator, so the cooking behaviour tree runs
against it unchanged:

    /switch_to_{position,navigation,trajectory}_mode  std_srvs/Trigger            (stretch_driver)
    /navigate_to_pose                                  nav2_msgs/NavigateToPose     (Nav2)
    /stretch_controller/follow_joint_trajectory        control_msgs/FollowJointTrajectory, head joints
    /detect_floor_objects_3d                           DetectFloorObjects3D         (floor_object_detector)
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
from std_msgs.msg import String
from std_srvs.srv import Trigger

from stretch_sim.events import INFO
from stretch_sim.geometry import Pose2D
from stretch_sim.operations import OpStatus
from stretch_sim.robot import DRIVER_MODES
from stretch_sim.simulator import KinematicSimulator

HEAD_JOINTS = ("joint_head_pan", "joint_head_tilt")
# Everything the sim may offer: what the robot offers with bringup_stretch.sh running. The
# sim must not offer anything else, or a BT that works against it could fail on the robot.
ROBOT_INTERFACES = frozenset({
    "/switch_to_position_mode", "/switch_to_navigation_mode", "/switch_to_trajectory_mode",
    "/switch_to_gamepad_mode",                                      # stretch_driver
    "/navigate_to_pose",                                            # Nav2
    "/stretch_controller/follow_joint_trajectory",                  # stretch_driver
    "/detect_floor_objects_3d",                                     # floor_object_detector
    "/grasp_carry", "/topdown_grasp",                               # grasp servers
    "/handover",                                                    # handover (run_once:=false)
    "/stretch/joint_states",                                        # stretch_driver
    "/camera/color/image_raw/compressed", "/camera/aligned_depth_to_color/image_raw/compressedDepth",
    "/camera/color/camera_info",                                    # realsense
})
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

        NavigateToPose = _optional("nav2_msgs.action", "NavigateToPose")
        FollowJointTrajectory = _optional("control_msgs.action", "FollowJointTrajectory")
        DetectFloorObjects3D = _optional("stretch_floor_perception.srv", "DetectFloorObjects3D")
        GraspPoint = _optional("stretch_floor_perception.srv", "GraspPoint")
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
    HEAD
    '''
    def _follow_trajectory(self, goal_handle):
        FollowJointTrajectory = self._FollowJointTrajectory
        traj = goal_handle.request.trajectory
        result = FollowJointTrajectory.Result()
        unsupported = [j for j in traj.joint_names if j not in HEAD_JOINTS]
        if unsupported or not traj.points:
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS if unsupported else FollowJointTrajectory.Result.INVALID_GOAL
            result.error_string = (f"the sim only moves {list(HEAD_JOINTS)}, not {unsupported}" if unsupported
                                   else "trajectory has no points")
            self.sim.emit(INFO, "ros", f"follow_joint_trajectory rejected: {result.error_string}")
            goal_handle.abort()
            return result

        last = traj.points[-1]
        target = dict(zip(traj.joint_names, last.positions))
        op = self.sim.move_head(target.get("joint_head_pan", self.sim.robot.head_pan),
                                target.get("joint_head_tilt", self.sim.robot.head_tilt),
                                max(_duration_s(last.time_from_start), 0.1))
        if not self._wait(op, goal_handle):
            goal_handle.canceled()
            return result
        if op.status == OpStatus.SUCCEEDED:
            result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
            goal_handle.succeed()
        else:
            result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
            result.error_string = op.message
            goal_handle.abort()
        return result

    '''
    PERCEPTION, GRASPING
    '''
    def _detect(self, req, res):
        stamp = self.get_clock().now().to_msg()
        res.detections.header.frame_id = "base_link"
        res.detections.header.stamp = stamp
        for d in self.sim.detect():
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
            res.detections.detections.append(m)
        return res

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
        if not self._bt_reset.wait_for_service(timeout_sec=min(timeout_s, 2.0)):
            return False, f"no behaviour tree offers {BT_RESET_SERVICE}"
        done = threading.Event()
        future = self._bt_reset.call_async(Trigger.Request())
        future.add_done_callback(lambda f: done.set())
        if not done.wait(timeout_s):
            self._bt_reset.remove_pending_request(future)
            return False, f"no answer from {BT_RESET_SERVICE} within {timeout_s:.0f} s"
        response = future.result()
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
                      "joint_arm": r.arm_extension, "joint_gripper_finger_left": finger,
                      "joint_gripper_finger_right": finger}
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(values)
        msg.position = [float(v) for v in values.values()]
        self.joint_pub.publish(msg)

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
