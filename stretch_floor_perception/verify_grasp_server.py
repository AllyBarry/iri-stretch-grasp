#!/usr/bin/env python3
"""verify_grasp_server.py: ROS 2 service that checks what the gripper is holding, using the trained grasp verifier.

  ros2 run stretch_floor_perception verify_grasp_server.py
  ros2 run stretch_floor_perception verify_grasp_server.py --ros-args -p move_to_check_pose:=false   # already posed
  ros2 run stretch_floor_perception verify_grasp_server.py --ros-args -p model:=$HOME/stretch_user/models/grasp_binary.pt
  ros2 run stretch_floor_perception verify_grasp_server.py --ros-args -p restore_pose:=false   # stay in the check pose
  ros2 run stretch_floor_perception verify_grasp_server.py --ros-args -p keep_lift:=true   # check at the current height

  ros2 service call /verify_grasp std_srvs/srv/Trigger {}

Response (std_srvs/Trigger):
  success = True only if something is in the gripper, the frames agree, and the model is confident
  message = JSON, e.g.
    {"holding": true, "item": "lemon", "label": "lemon", "conf": 0.97, "agree": 1.0,
     "votes": {"lemon": 5}, "frames": 5, "uncertain": false, "classes": ["balsamic", ..., "salt"]}
  success False + "uncertain": true  -> frames disagreed or low confidence: look again / retry the check
  success False + "label": "empty"   -> missed grasp
  "error" key                        -> the check itself failed (no camera, arm move rejected, ...)
  "classes"                          -> every label the model can give, so callers can tell an item it
                                        was not trained on from a wrong item
The same JSON is also published on /grasp_verifier/result (std_msgs/String).

Each call: switch to position mode if needed -> move lift/arm/wrist/head to the check pose saved in
~/dataset/grasp/preset.json (the pose the training images were taken in) -> wait to settle -> classify
N fresh head-camera frames, rotated exactly like the training images -> majority vote -> move back to the
previous arm/head pose and mode. The gripper and the base are never commanded.

With keep_lift, the lift stays where it is (for example just above the table after a pick) and
only the arm, wrist and head move to the check pose. The head then tilts by however much the
lift differs from the check pose, so the gripper is where it was in the training images.
"""
import json
import math
import os
import threading
import time
from collections import Counter

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

POSE_JOINTS = ["joint_lift", "wrist_extension", "joint_wrist_yaw", "joint_wrist_pitch",
               "joint_wrist_roll", "joint_head_pan", "joint_head_tilt"]
DEFAULT_CHECK_POSE = {"joint_lift": 0.8, "wrist_extension": 0.0, "joint_wrist_yaw": 0.0,
                      "joint_wrist_pitch": 0.0, "joint_wrist_roll": 0.0,
                      "joint_head_pan": -1.57, "joint_head_tilt": -0.6}
NOT_HOLDING = ("empty", "no gripper found")
HEAD_TILT_RANGE = (-1.53, 0.79)   # joint_head_tilt limits


class GraspVerifier(Node):
    def __init__(self):
        super().__init__("grasp_verifier")
        p = self.declare_parameter
        self.model_path = os.path.expanduser(p("model", "~/stretch_user/models/grasp_items.pt").value)
        topic = p("image_topic", "/camera/color/image_raw/compressed").value
        self.rotate = bool(p("rotate", True).value)            # training frames were rotated 90 deg clockwise
        self.n_frames = int(p("frames", 5).value)
        self.min_conf = float(p("min_conf", 0.6).value)        # mean confidence of the winning class
        self.min_agree = float(p("min_agree", 0.6).value)      # fraction of frames that must agree
        self.move_pose = bool(p("move_to_check_pose", True).value)
        self.restore = bool(p("restore_pose", True).value)
        self.keep_lift = bool(p("keep_lift", False).value)    # check at the current lift height
        self.move_time = float(p("move_time", 3.0).value)
        self.settle = float(p("settle", 0.7).value)
        self.preset_file = os.path.expanduser(p("preset_file", "~/dataset/grasp/preset.json").value)

        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        self.model(np.zeros((640, 480, 3), np.uint8), verbose=False)  # warm up so the first call is fast
        self.classes = [str(c) for c in self.model.names.values()]
        self.get_logger().info(f"loaded {self.model_path}, classes: {self.classes}")

        cb = ReentrantCallbackGroup()
        self.lock = threading.Lock()
        self.busy = threading.Lock()
        self.msg, self.msg_t = None, 0.0
        self.joints, self.mode = {}, ""
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(CompressedImage if "compressed" in topic else Image, topic,
                                 self.on_image, qos, callback_group=cb)
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10, callback_group=cb)
        self.create_subscription(String, "/mode", self.on_mode, 10, callback_group=cb)
        self.traj = ActionClient(self, FollowJointTrajectory, "/stretch_controller/follow_joint_trajectory",
                                 callback_group=cb)
        self.mode_cli = {m: self.create_client(Trigger, f"/switch_to_{m}_mode", callback_group=cb)
                         for m in ("position", "navigation")}
        self.pub = self.create_publisher(String, "/grasp_verifier/result", 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.create_service(Trigger, "/verify_grasp", self.on_verify, callback_group=cb)
        self.get_logger().info(f"ready: ros2 service call /verify_grasp std_srvs/srv/Trigger {{}}   (image topic {topic})")

    # ------------------------------------------------------------ subscriptions
    def on_image(self, msg):
        with self.lock:
            self.msg, self.msg_t = msg, time.time()

    def on_joints(self, msg):
        with self.lock:
            for i, n in enumerate(msg.name):
                if i < len(msg.position):
                    self.joints[n] = float(msg.position[i])

    def on_mode(self, msg):
        with self.lock:
            self.mode = msg.data

    # ------------------------------------------------------------ helpers
    def decode(self, msg):
        if isinstance(msg, CompressedImage):
            img = cv2.imdecode(np.frombuffer(bytes(msg.data), np.uint8), cv2.IMREAD_COLOR)
        else:
            arr = np.frombuffer(bytes(msg.data), np.uint8)
            img = arr.reshape(msg.height, msg.step // 3, 3)[:, : msg.width]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else img.copy()
        if img is not None and self.rotate:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img

    @staticmethod
    def wait(fut, timeout):
        end = time.time() + timeout
        while not fut.done():
            if time.time() > end:
                return None
            time.sleep(0.02)
        return fut.result()

    def set_mode(self, m):
        c = self.mode_cli[m]
        if not c.wait_for_service(timeout_sec=2.0):
            return f"/switch_to_{m}_mode not available (is stretch_driver running?)"
        if self.wait(c.call_async(Trigger.Request()), 10.0) is None:
            return f"switch to {m} mode timed out"
        end = time.time() + 3.0
        while time.time() < end:
            with self.lock:
                if not self.mode or self.mode == m:
                    break
            time.sleep(0.05)
        return ""

    def move(self, targets, secs):
        if not self.traj.wait_for_server(timeout_sec=2.0):
            return "trajectory action server not available"
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(targets)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in targets.values()]
        pt.time_from_start = Duration(sec=int(secs), nanosec=int((secs % 1) * 1e9))
        goal.trajectory.points = [pt]
        h = self.wait(self.traj.send_goal_async(goal), 5.0)
        if h is None or not h.accepted:
            return "arm move rejected (position mode?)"
        r = self.wait(h.get_result_async(), secs + 10.0)
        if r is None:
            return "arm move timed out"
        if r.result.error_code != 0:
            return f"arm move failed (error {r.result.error_code})"
        return ""

    def check_pose(self):
        pose = dict(DEFAULT_CHECK_POSE)
        try:
            with open(self.preset_file) as f:
                pose.update(json.load(f).get("check_pose", {}))
        except (OSError, ValueError):
            self.get_logger().warn(f"no {self.preset_file}, using the default check pose")
        return {j: pose[j] for j in POSE_JOINTS}

    def tilt_for_lift(self, tilt, trained_lift):
        """Head tilt that keeps the gripper where it was in the training images, which were taken
        with the lift at trained_lift: the gripper is now higher or lower by the lift difference,
        so tilt by the change in angle from the head to it."""
        try:
            head = self.tf_buffer.lookup_transform("base_link", "link_head_tilt", Time()).transform.translation
            grip = self.tf_buffer.lookup_transform("base_link", "link_grasp_center", Time()).transform.translation
        except Exception as e:  # TF not available: keep the trained tilt
            self.get_logger().warn(f"no head/gripper transform ({e}), keeping the trained head tilt")
            return tilt
        with self.lock:
            lift = self.joints.get("joint_lift")
        if lift is None:
            return tilt
        horiz = math.hypot(grip.x - head.x, grip.y - head.y)
        dz = grip.z - head.z
        tilt += math.atan2(dz, horiz) - math.atan2(dz + (trained_lift - lift), horiz)
        return min(max(tilt, HEAD_TILT_RANGE[0]), HEAD_TILT_RANGE[1])

    def grab(self, n, after):
        frames, last = [], after
        end = time.time() + 3.0 + n * 0.5
        while len(frames) < n and time.time() < end:
            with self.lock:
                msg, t = self.msg, self.msg_t
            if msg is not None and t > last:
                last = t
                img = self.decode(msg)
                if img is not None:
                    frames.append(img)
                time.sleep(0.1)   # spread frames out a little
            else:
                time.sleep(0.02)
        return frames

    def classify(self, img):
        r = self.model(img, verbose=False)[0]
        if r.probs is not None:   # classifier
            return str(r.names[int(r.probs.top1)]), float(r.probs.top1conf)
        best = ("no gripper found", 0.0)   # detector trained on gripper boxes
        for c, q in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if q > best[1]:
                best = (str(r.names[int(c)]).replace("held_", "").replace("gripper_empty", "empty"), float(q))
        return best

    def vote(self, preds):
        counts = Counter(n for n, _ in preds)
        top, k = counts.most_common(1)[0]
        conf = float(np.mean([c for n, c in preds if n == top]))
        agree = k / len(preds)
        uncertain = agree < self.min_agree or conf < self.min_conf
        return {"holding": top not in NOT_HOLDING and not uncertain,
                "item": "" if top in NOT_HOLDING else top, "label": top,
                "conf": round(conf, 3), "agree": round(agree, 2),
                "votes": dict(counts), "frames": len(preds), "uncertain": uncertain}

    # ------------------------------------------------------------ service
    def verify(self):
        with self.lock:
            prev_mode = self.mode
            prev_pose = {j: self.joints[j] for j in POSE_JOINTS if j in self.joints}
        result = {"holding": False, "item": "", "label": "", "conf": 0.0, "votes": {}, "frames": 0}
        notes, moved, switched = [], False, False
        try:
            if self.move_pose:
                if prev_mode != "position":
                    err = self.set_mode("position")
                    if err:
                        result["error"] = err
                        return result
                    switched = True
                moved = True
                pose = self.check_pose()
                trained_lift = pose.pop("joint_lift") if self.keep_lift else None
                err = self.move(pose, self.move_time)
                if err:
                    result["error"] = err
                    return result
                if trained_lift is not None:
                    tilt = self.tilt_for_lift(pose["joint_head_tilt"], trained_lift)
                    if abs(tilt - pose["joint_head_tilt"]) > 0.01:
                        err = self.move({"joint_head_tilt": tilt}, 1.0)
                        if err:
                            result["error"] = err
                            return result
                time.sleep(self.settle)
            frames = self.grab(self.n_frames, time.time())
            if not frames:
                result["error"] = "no camera frames (is the camera running?)"
                return result
            result.update(self.vote([self.classify(f) for f in frames]))
            return result
        finally:
            if moved and self.restore:
                if len(prev_pose) == len(POSE_JOINTS):
                    err = self.move(prev_pose, self.move_time)
                    if err:
                        notes.append(f"move back failed: {err}")
                else:
                    notes.append("no joint states, did not move back")
            if switched and prev_mode == "navigation":
                err = self.set_mode("navigation")
                if err:
                    notes.append(err)
            if notes:
                result["note"] = "; ".join(notes)

    def on_verify(self, req, res):
        if not self.busy.acquire(blocking=False):
            res.success, res.message = False, json.dumps({"error": "busy with another check"})
            return res
        t0 = time.time()
        try:
            result = self.verify()
        except Exception as e:  # never kill the node on a bad call
            result = {"holding": False, "item": "", "error": repr(e)}
        finally:
            self.busy.release()
        result["seconds"] = round(time.time() - t0, 2)
        result["classes"] = self.classes
        res.success = bool(result.get("holding"))
        res.message = json.dumps(result)
        self.pub.publish(String(data=res.message))
        self.get_logger().info(f"/verify_grasp -> {res.message}")
        return res


def main():
    rclpy.init()
    node = GraspVerifier()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
