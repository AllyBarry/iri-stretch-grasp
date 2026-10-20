#!/usr/bin/env python3

from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState
from visualization_msgs.msg import Marker

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
import rclpy.logging

import time
import threading
import numpy as np
import os
import cv2
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time

import hello_helpers.hello_misc as hm
import stretch_funmap.navigate as nv
import stretch_funmap.manipulation_planning as mp
from stretch_floor_perception.srv import DetectObjects2D, GraspPoint


class DebugMoveCancelled(RuntimeError):
    pass


class GraspPointNode(hm.HelloNode):

    def __init__(self):
        hm.HelloNode.__init__(self)
        self.rate = 10.0
        self.joint_states = None
        self.joint_states_lock = threading.Lock()
        self.move_base = nv.MoveBase(self)
        self.wrist_position = None
        self.lift_position = None
        self.debug_directory = None
        self.assumed_width_m = 0.05
        self.scan_time_s = 2.0
        self.debug_mode = False
        self.debug_marker_pub = None
        self.debug_prompt_lock = threading.Lock()
        self.carry_wrist_pitch_rad = -0.3
        self.carry_wrist_yaw_offset_rad = 0.5
        self.carry_wrist_roll_rad = 1.5708
        self.look_at_gripper = True
        self.stow_before_grasp = False
        self.verify_grasp_visually = True
        self.color_lock = threading.Lock()
        self.latest_color = None
        self.latest_color_info = None
        self.cv_bridge = CvBridge()

    def joint_states_callback(self, joint_states):
        with self.joint_states_lock:
            self.joint_states = joint_states
        wrist_position, _, _ = hm.get_wrist_state(joint_states)
        self.wrist_position = wrist_position
        lift_position, _, _ = hm.get_lift_state(joint_states)
        self.lift_position = lift_position

    def drive(self, forward_m):
        if self.dryrun:
            return

        tolerance_distance_m = 0.005
        if forward_m > 0:
            self.move_base.forward(forward_m, detect_obstacles=False, tolerance_distance_m=tolerance_distance_m, max_forward_attempts=1)
        else:
            self.move_base.backward(forward_m, detect_obstacles=False, tolerance_distance_m=tolerance_distance_m)

    def look_at_surface(self, scan_time_s=None):
        if self.dryrun:
            return None

        manip = mp.ManipulationView(self.tf2_buffer, self.debug_directory, self.get_tool())
        manip.move_head(self.debug_move_to_pose)
        time.sleep(0.02)

        if scan_time_s is None:
            manip.update(self.point_cloud, self.tf2_buffer)
        else:
            start_time_s = time.time()
            while (time.time() - start_time_s) < scan_time_s:
                manip.update(self.point_cloud, self.tf2_buffer)

        if self.debug_directory is not None:
            dirname = self.debug_directory + 'grasp_carry/'
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            filename = 'look_at_surface_' + hm.create_time_string()
            manip.save_scan(dirname + filename)

        return manip

    def _get_gripper_tip_in_map(self):
        tip_to_map_mat, _ = hm.get_p1_to_p2_matrix('link_grasp_center', 'map', self.tf2_buffer)
        if tip_to_map_mat is None:
            self.logger.warn('Unable to compute the gripper tip pose in map for debug visualization.')
            return None
        return tip_to_map_mat[:3, 3].copy()

    def _get_base_axes_in_map(self):
        base_to_map_mat, _ = hm.get_p1_to_p2_matrix('base_link', 'map', self.tf2_buffer)
        if base_to_map_mat is None:
            self.logger.warn('Unable to compute the base pose in map for debug visualization.')
            return None
        return (
            base_to_map_mat[:3, 0].copy(),
            base_to_map_mat[:3, 1].copy(),
            base_to_map_mat[:3, 2].copy(),
        )

    def _compute_translation_preview(self, pose=None, forward_m=None):
        if self.dryrun:
            return None

        tip_xyz = self._get_gripper_tip_in_map()
        axes = self._get_base_axes_in_map()
        if (tip_xyz is None) or (axes is None):
            return None

        base_x_axis, base_y_axis, base_z_axis = axes
        translation_xyz = np.zeros(3, dtype=float)

        if forward_m is not None:
            translation_xyz += float(forward_m) * base_x_axis

        if pose is not None:
            if 'translate_mobile_base' in pose:
                translation_xyz += float(pose['translate_mobile_base']) * base_x_axis

            if ('wrist_extension' in pose) and (self.wrist_position is not None):
                delta_extension_m = float(pose['wrist_extension']) - float(self.wrist_position)
                translation_xyz += delta_extension_m * (-base_y_axis)

            if ('joint_lift' in pose) and (self.lift_position is not None):
                delta_lift_m = float(pose['joint_lift']) - float(self.lift_position)
                translation_xyz += delta_lift_m * base_z_axis

        if np.linalg.norm(translation_xyz) < 1e-4:
            return None

        return tip_xyz, tip_xyz + translation_xyz

    def _publish_translation_marker(self, start_xyz, end_xyz):
        if self.debug_marker_pub is None:
            return

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'grasp_carry_debug'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.scale.x = 0.015
        marker.scale.y = 0.03
        marker.scale.z = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.2
        marker.color.a = 0.9
        marker.points = [
            Point(x=float(start_xyz[0]), y=float(start_xyz[1]), z=float(start_xyz[2])),
            Point(x=float(end_xyz[0]), y=float(end_xyz[1]), z=float(end_xyz[2])),
        ]
        self.debug_marker_pub.publish(marker)

    def _clear_translation_marker(self):
        if self.debug_marker_pub is None:
            return

        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'grasp_carry_debug'
        marker.id = 0
        marker.action = Marker.DELETE
        self.debug_marker_pub.publish(marker)

    def _describe_move(self, pose=None, description=None, forward_m=None):
        if description is not None:
            return description
        if forward_m is not None:
            return f'translate mobile base by {forward_m:.3f} m'
        if pose is None:
            return 'execute robot move'
        return 'move to pose ' + ', '.join(f'{joint}={value:.3f}' for joint, value in pose.items())

    def _confirm_motion(self, description, pose=None, forward_m=None):
        if (not self.debug_mode) or self.dryrun:
            return

        preview = self._compute_translation_preview(pose=pose, forward_m=forward_m)
        if preview is not None:
            self._publish_translation_marker(*preview)
        else:
            self._clear_translation_marker()

        prompt = f'{description}. Proceed? [y/N]: '
        try:
            with self.debug_prompt_lock:
                answer = input(prompt).strip().lower()
        except EOFError:
            answer = ''

        if answer not in ('y', 'yes'):
            raise DebugMoveCancelled(f'User declined motion: {description}')

    def debug_move_to_pose(self, pose, blocking=True, custom_contact_thresholds=False, duration=2.0, description=None):
        self._confirm_motion(self._describe_move(pose=pose, description=description), pose=pose)
        return self.move_to_pose(
            pose,
            blocking=blocking,
            custom_contact_thresholds=custom_contact_thresholds,
            duration=duration,
        )

    def debug_drive(self, forward_m, description=None):
        self._confirm_motion(self._describe_move(description=description, forward_m=forward_m), forward_m=forward_m)
        self.drive(forward_m)

    def _point_head_at_gripper_tip(self):
        if self.dryrun:
            return

        # Aim the head camera at the gripper tip by computing pan/tilt from the
        # tip position relative to the head tilt joint, expressed in base_link.
        tip_to_base_mat, _ = hm.get_p1_to_p2_matrix('link_grasp_center', 'base_link', self.tf2_buffer)
        head_to_base_mat, _ = hm.get_p1_to_p2_matrix('link_head_tilt', 'base_link', self.tf2_buffer)
        if (tip_to_base_mat is None) or (head_to_base_mat is None):
            self.logger.warn('Unable to compute the gripper tip relative to the head, so not aiming the camera.')
            return

        dx, dy, dz = tip_to_base_mat[:3, 3] - head_to_base_mat[:3, 3]
        pan = float(np.arctan2(dy, dx))
        tilt = float(np.arctan2(dz, np.hypot(dx, dy)))
        self.logger.info(f'Point the head camera at the gripper tip (pan={pan:.3f}, tilt={tilt:.3f}).')
        self.debug_move_to_pose({'joint_head_pan': pan, 'joint_head_tilt': tilt})

    def _get_joint_position(self, joint_name):
        with self.joint_states_lock:
            joint_states = self.joint_states
        if (joint_states is None) or (joint_name not in joint_states.name):
            return None
        return joint_states.position[joint_states.name.index(joint_name)]

    def _present_gripper_to_camera(self):
        # Swing the gripper sideways (positive yaw is toward the front of the robot)
        # so the object is not hidden behind the arm from the head camera's view.
        pose = {}
        wrist_yaw = self._get_joint_position('joint_wrist_yaw')
        if wrist_yaw is None:
            self.logger.warn('Wrist yaw unavailable, so not turning the gripper sideways.')
        elif self.carry_wrist_yaw_offset_rad != 0.0:
            pose['joint_wrist_yaw'] = float(np.clip(wrist_yaw + self.carry_wrist_yaw_offset_rad, -1.75, 4.0))

        if self.get_tool() == "tool_stretch_dex_wrist":
            pose['joint_wrist_pitch'] = self.carry_wrist_pitch_rad
            # Rolling the wrist turns the held object onto its side, so the camera
            # sees its length instead of looking down onto the fingers.
            pose['joint_wrist_roll'] = self.carry_wrist_roll_rad

        if pose:
            self.logger.info('Turn the gripper to the side and roll the object so it is visible to the head camera.')
            self.debug_move_to_pose(pose)

        if self.look_at_gripper:
            self._point_head_at_gripper_tip()

    # ------------------------------------------------------------------
    # Grasp check: detect the expected label in the gripper and confirm it is
    # no longer at the pick location, using the detector's detect_objects_2d.
    # ------------------------------------------------------------------

    def _transform_point(self, point_xyz, source_frame, target_frame):
        if source_frame == target_frame:
            return np.asarray(point_xyz, dtype=float)
        transform_mat, _ = hm.get_p1_to_p2_matrix(source_frame, target_frame, self.tf2_buffer)
        if transform_mat is None:
            return None
        return transform_mat[:3, :3] @ np.asarray(point_xyz, dtype=float) + transform_mat[:3, 3]

    def _point_head_at(self, point_base, what='the target'):
        if self.dryrun:
            return
        head_to_base_mat, _ = hm.get_p1_to_p2_matrix('link_head_tilt', 'base_link', self.tf2_buffer)
        if head_to_base_mat is None:
            self.logger.warn(f'Unable to locate the head, so not aiming the camera at {what}.')
            return
        dx, dy, dz = np.asarray(point_base, dtype=float) - head_to_base_mat[:3, 3]
        pan = float(np.arctan2(dy, dx))
        tilt = float(np.arctan2(dz, np.hypot(dx, dy)))
        self.logger.info(f'Point the head camera at {what} (pan={pan:.3f}, tilt={tilt:.3f}).')
        self.debug_move_to_pose({'joint_head_pan': pan, 'joint_head_tilt': tilt})

    def _color_callback(self, msg):
        with self.color_lock:
            self.latest_color = msg

    def _color_info_callback(self, msg):
        with self.color_lock:
            self.latest_color_info = msg

    def _setup_grasp_check(self):
        for name, default in (
            ('carry_wrist_roll_rad', 1.5708),
            ('color_topic', '/camera/color/image_raw/compressed'),
            ('color_info_topic', '/camera/color/camera_info'),
            ('detection_service', '/detect_objects_2d'),
            ('grasp_check_settle_s', 1.0),
            ('grasp_check_frames', 3),
            ('grasp_check_min_frames', 2),
            ('grasp_check_min_score', 0.25),
            ('grasp_check_roi_radius_m', 0.08),
            ('grasp_check_timeout_s', 5.0),
            ('check_pick_location', True),
            ('pick_check_radius_m', 0.06),
            ('accept_pick_location_clear', False),
        ):
            self._declare_parameter_once(name, default)
        self.carry_wrist_roll_rad = float(self.get_parameter('carry_wrist_roll_rad').value)
        self.grasp_check_settle_s = float(self.get_parameter('grasp_check_settle_s').value)
        self.grasp_check_frames = int(self.get_parameter('grasp_check_frames').value)
        self.grasp_check_min_frames = int(self.get_parameter('grasp_check_min_frames').value)
        self.grasp_check_min_score = float(self.get_parameter('grasp_check_min_score').value)
        self.grasp_check_roi_radius_m = float(self.get_parameter('grasp_check_roi_radius_m').value)
        self.grasp_check_timeout_s = float(self.get_parameter('grasp_check_timeout_s').value)
        self.check_pick_location = bool(self.get_parameter('check_pick_location').value)
        self.pick_check_radius_m = float(self.get_parameter('pick_check_radius_m').value)
        self.accept_pick_location_clear = bool(self.get_parameter('accept_pick_location_clear').value)

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.color_subscriber = self.create_subscription(
            CompressedImage, self.get_parameter('color_topic').value, self._color_callback, qos,
            callback_group=self.callback_group)
        self.color_info_subscriber = self.create_subscription(
            CameraInfo, self.get_parameter('color_info_topic').value, self._color_info_callback, qos,
            callback_group=self.callback_group)
        self.detection_client = self.create_client(
            DetectObjects2D, self.get_parameter('detection_service').value, callback_group=self.callback_group)
        self.grasp_check_image_pub = self.create_publisher(Image, f'/{self.get_name()}/grasp_check_image', 1)

    @staticmethod
    def _normalize_label(label):
        return str(label).strip().lower().replace('_', ' ')

    def _wait_for_fresh_color(self, timeout_s=3.0):
        start_ns = self.get_clock().now().nanoseconds
        deadline_s = time.time() + timeout_s
        while time.time() < deadline_s:
            with self.color_lock:
                color_msg, info_msg = self.latest_color, self.latest_color_info
            if (color_msg is not None) and (info_msg is not None) and \
                    Time.from_msg(color_msg.header.stamp).nanoseconds > start_ns:
                return color_msg, info_msg
            time.sleep(0.05)
        return None, None

    def _detect_2d(self, color_msg):
        if not self.detection_client.wait_for_service(timeout_sec=1.0):
            self.logger.error(f'Detection service {self.detection_client.srv_name} is unavailable.')
            return None
        request = DetectObjects2D.Request()
        request.color_msg = color_msg
        future = self.detection_client.call_async(request)
        deadline_s = time.time() + self.grasp_check_timeout_s
        while not future.done() and time.time() < deadline_s:
            time.sleep(0.02)
        if not future.done() or future.result() is None:
            self.logger.error('Detection service did not respond in time.')
            return None
        detections = []
        for det in future.result().detections.detections:
            if not det.results:
                continue
            best = max(det.results, key=lambda r: r.hypothesis.score)
            cx, cy = det.bbox.center.position.x, det.bbox.center.position.y
            box = (cx - det.bbox.size_x / 2.0, cy - det.bbox.size_y / 2.0,
                   cx + det.bbox.size_x / 2.0, cy + det.bbox.size_y / 2.0)
            detections.append((best.hypothesis.class_id, float(best.hypothesis.score), box))
        return detections

    def _project_to_image(self, point_xyz, source_frame, info_msg):
        point_cam = self._transform_point(point_xyz, source_frame, info_msg.header.frame_id)
        if point_cam is None or point_cam[2] <= 0.05:
            return None
        k = np.array(info_msg.k, dtype=float).reshape(3, 3)
        u, v, _ = k @ (point_cam / point_cam[2])
        return float(u), float(v), float(point_cam[2])

    def _image_region_around(self, points, source_frame, info_msg, radius_m):
        # Pixel rectangle covering the projected points, padded by radius_m at their depth.
        projected = [self._project_to_image(p, source_frame, info_msg) for p in points]
        projected = [p for p in projected if p is not None]
        if not projected:
            return None
        fx = float(info_msg.k[0])
        pad_px = max(fx * radius_m / z for _, _, z in projected)
        us = [u for u, _, _ in projected]
        vs = [v for _, v, _ in projected]
        width, height = info_msg.width, info_msg.height
        x1, y1 = max(0.0, min(us) - pad_px), max(0.0, min(vs) - pad_px)
        x2, y2 = min(width - 1.0, max(us) + pad_px), min(height - 1.0, max(vs) + pad_px)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2), (projected[0][0], projected[0][1])

    def _gripper_region(self, info_msg):
        # The grasp center and both fingertips, padded by the expected object size.
        points = []
        for frame in ('link_grasp_center', 'link_gripper_fingertip_left', 'link_gripper_fingertip_right'):
            point = self._transform_point(np.zeros(3), frame, 'base_link')
            if point is not None:
                points.append(point)
        if not points:
            return None
        return self._image_region_around(points, 'base_link', info_msg, self.grasp_check_roi_radius_m)

    @staticmethod
    def _box_overlaps_region(box, region, center_px=None):
        x1, y1, x2, y2 = box
        rx1, ry1, rx2, ry2 = region
        inter = max(0.0, min(x2, rx2) - max(x1, rx1)) * max(0.0, min(y2, ry2) - max(y1, ry1))
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        contains_center = (center_px is not None) and (x1 <= center_px[0] <= x2) and (y1 <= center_px[1] <= y2)
        return contains_center or (inter / area) >= 0.3

    def _publish_grasp_check_image(self, color_msg, region, detections, expected, verdict):
        try:
            image = self.cv_bridge.compressed_imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        except Exception as exc:
            self.logger.warn(f'Unable to decode the colour image for the grasp check image: {exc}')
            return
        height = image.shape[0]
        labels = []
        if region is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in region]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 1)
        for label, score, box in detections:
            match = bool(expected) and self._normalize_label(label) == expected
            color = (0, 255, 0) if match else (0, 0, 255)
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            labels.append((f'{label} {score:.2f}', (x1, y2), color))
        # The head camera is mounted sideways, so rotate to upright before adding text.
        # ROTATE_90_CLOCKWISE maps (x, y) to (height - 1 - y, x).
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        for text, (x, y), color in labels:
            cv2.putText(image, text, (max(0, height - 1 - y), max(10, x - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        cv2.putText(image, verdict, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        msg = self.cv_bridge.cv2_to_imgmsg(image, encoding='bgr8')
        msg.header = color_msg.header
        self.grasp_check_image_pub.publish(msg)

    def _check_label_in_gripper(self, expected):
        """Returns (status, detail) with status 'confirmed', 'wrong', 'unseen' or 'unavailable'."""
        time.sleep(self.grasp_check_settle_s)
        frames = 0
        hits = []
        others = {}
        last = None
        for _ in range(max(1, self.grasp_check_frames)):
            color_msg, info_msg = self._wait_for_fresh_color()
            if color_msg is None:
                break
            region = self._gripper_region(info_msg)
            detections = self._detect_2d(color_msg)
            if region is None or detections is None:
                break
            frames += 1
            box_region, center_px = region
            in_gripper = [d for d in detections
                          if d[1] >= self.grasp_check_min_score
                          and self._box_overlaps_region(d[2], box_region, center_px)]
            matched = [d for d in in_gripper if (not expected) or self._normalize_label(d[0]) == expected]
            if matched:
                hits.append(max(matched, key=lambda d: d[1]))
            for label, score, _ in in_gripper:
                if expected and self._normalize_label(label) != expected:
                    others[label] = max(score, others.get(label, 0.0))
            last = (color_msg, box_region, detections)
            self.logger.info(
                'Grasp check (in gripper): '
                + (', '.join(f'{d[0]} {d[1]:.2f}' for d in in_gripper) or 'nothing detected in the gripper region'))

        if frames == 0:
            return 'unavailable', 'no camera image or detections available to check the gripper'

        if len(hits) >= min(self.grasp_check_min_frames, frames):
            best = max(hits, key=lambda d: d[1])
            status = 'confirmed'
            detail = f"'{best[0]}' ({best[1]:.2f}) detected in the gripper in {len(hits)}/{frames} frames"
        elif others:
            label, score = max(others.items(), key=lambda kv: kv[1])
            status = 'wrong'
            detail = f"holding '{label}' ({score:.2f}), expected '{expected}'"
        else:
            status = 'unseen'
            detail = (f"'{expected}' not detected in the gripper" if expected
                      else 'no object detected in the gripper') + f' ({len(hits)}/{frames} frames)'

        if last is not None:
            self._publish_grasp_check_image(last[0], last[1], last[2], expected, f'gripper: {detail}')
        return status, detail

    def _check_pick_location(self, expected, pick_odom):
        """Returns (status, detail) with status 'still_there', 'clear' or 'unavailable'."""
        pick_base = self._transform_point(pick_odom, 'odom', 'base_link')
        if pick_base is None:
            return 'unavailable', 'unable to locate the pick location'
        self._point_head_at(pick_base, 'the pick location')
        time.sleep(self.grasp_check_settle_s)
        color_msg, info_msg = self._wait_for_fresh_color()
        if color_msg is None:
            return 'unavailable', 'no camera image of the pick location'
        pick_region = self._image_region_around([pick_odom], 'odom', info_msg, self.pick_check_radius_m)
        if pick_region is None:
            return 'unavailable', 'the pick location is not in view'
        detections = self._detect_2d(color_msg)
        if detections is None:
            return 'unavailable', 'no detections of the pick location'

        # Ignore the held object if it happens to be in view as well.
        gripper = self._gripper_region(info_msg)
        found = []
        for label, score, box in detections:
            if self._normalize_label(label) != expected or score < self.grasp_check_min_score:
                continue
            if gripper is not None and self._box_overlaps_region(box, gripper[0], gripper[1]):
                continue
            if self._box_overlaps_region(box, pick_region[0], pick_region[1]):
                found.append((label, score, box))

        if found:
            best = max(found, key=lambda d: d[1])
            status, detail = 'still_there', f"'{best[0]}' ({best[1]:.2f}) still detected at the pick location"
        else:
            status, detail = 'clear', 'pick location clear'
        self.logger.info(f'Grasp check (pick location): {detail}')
        self._publish_grasp_check_image(color_msg, pick_region[0], detections, expected, f'pick location: {detail}')
        return status, detail

    def _verify_grasp(self, expected_label, pick_odom):
        """Returns (success, message) for the service response."""
        if self.dryrun:
            return True, 'dryrun, grasp check skipped'
        expected = self._normalize_label(expected_label) if expected_label else ''

        gripper_status, gripper_detail = self._check_label_in_gripper(expected)
        if gripper_status == 'wrong':
            return False, f'Wrong object: {gripper_detail}'

        details = [gripper_detail]
        pick_status = 'unavailable'
        if expected and self.check_pick_location and pick_odom is not None:
            pick_status, pick_detail = self._check_pick_location(expected, pick_odom)
            details.append(pick_detail)
            if self.look_at_gripper:
                self._point_head_at_gripper_tip()

        if pick_status == 'still_there':
            return False, 'Not confirmed: ' + '; '.join(reversed(details))
        if gripper_status == 'confirmed':
            return True, 'Confirmed: ' + '; '.join(details)
        if pick_status == 'clear' and self.accept_pick_location_clear:
            return True, 'Likely held (not identified in the gripper): ' + '; '.join(details)
        return False, 'Not confirmed: ' + '; '.join(details)

    def grasp_carry_service_callback(self, request, response):
        try:
            msg = request.target
            assumed_width_m = request.assumed_width_m if request.assumed_width_m > 0.0 else self.assumed_width_m

            if not msg.header.stamp.sec > 0:
                self.logger.error('No point cloud available yet, so unable to grasp.')
                response.success = False
                response.message = 'No point cloud available yet, so unable to grasp.'
                return response

            self.logger.info(f'Grasp point received: frame={msg.header.frame_id} x={msg.point.x:.3f} y={msg.point.y:.3f} z={msg.point.z:.3f}'
                             f" expected_label='{request.expected_label}'")

            # Remember where the object was, in odom, to check afterwards that it is gone.
            pick_odom = self._transform_point(
                [msg.point.x, msg.point.y, msg.point.z], msg.header.frame_id, 'odom')

            min_extension_m = 0.01
            if self.stow_before_grasp:
                self.stow_the_robot()
            else:
                # Retract the arm at its current height instead of stowing, so the
                # lift does not drop all the way down only to come back up.
                self.logger.info('Retract the arm before scanning.')
                self.debug_move_to_pose({'wrist_extension': min_extension_m})

            manip = self.look_at_surface(scan_time_s=self.scan_time_s)
            if manip is None:
                self.logger.error('Failed to build manipulation view.')
                response.success = False
                response.message = 'Failed to build manipulation view.'
                return response

            max_height_im = manip.max_height_im
            points_to_image_mat, _ = max_height_im.get_points_to_image_mat(
                msg.header.frame_id, self.tf2_buffer)
            if points_to_image_mat is None:
                self.logger.error('points_to_image_mat not found for frame: {msg.header.frame_id}')
                response.success = False
                response.message = 'points_to_image_mat not found for frame'
                return response

            point_xyz = np.array([msg.point.x, msg.point.y, msg.point.z, 1.0])
            self.logger.info(f"robot loc: {points_to_image_mat}")
            point_image_pix = np.matmul(points_to_image_mat, point_xyz)[:3]
            i_x, i_y, i_z = point_image_pix
            self.logger.info(f"point image pix: x={i_x:.3f} y={i_y:.3f} z={i_z:.3f}")

            h, w = max_height_im.image.shape
            if not ((i_x >= 0) and (i_y >= 0) and (i_x < w) and (i_y < h)):
                self.logger.error('Target point is outside max_height_image bounds')
                response.success = False
                response.message = 'Target point is outside max_height_image bounds'
                return response

            grasp_target = {
                'location_xy_pix': np.array([i_x, i_y]),
                'location_z_pix': i_z,
                'elongated': False,
                'long_axis_pix': None,
                'width_m': assumed_width_m,
            }

            max_lift_m = 1.09
            min_lift_m = 0.07
            max_extension_m = 0.5

            pregrasp_lift_m = manip.get_pregrasp_lift(grasp_target, self.tf2_buffer)
            if self.get_tool() == "tool_stretch_dex_wrist":
                pregrasp_lift_m += 0.02
            if self.lift_position is None:
                self.logger.error('lift position unavailable')
                response.success = False
                response.message = 'lift position unavailable'
                return response

            pregrasp_yaw = manip.get_pregrasp_yaw(grasp_target, self.tf2_buffer)

            # Go straight to the pregrasp height with the arm retracted, so the arm
            # only moves along the mast and does not sweep through the table.
            self.logger.info('Raise/Lower tool to pregrasp height.')
            lift_to_pregrasp_m = max(self.lift_position + pregrasp_lift_m, min_lift_m)
            lift_to_pregrasp_m = min(lift_to_pregrasp_m, max_lift_m)
            self.debug_move_to_pose({'joint_lift': lift_to_pregrasp_m})

            if self.get_tool() == "tool_stretch_dex_wrist":
                self.logger.info('Rotate pitch/roll for grasping.')
                self.debug_move_to_pose({'joint_wrist_pitch': -0.3, 'joint_wrist_roll': 0.0})

            # Near the floor the gripper can hit the base while rotating out of its
            # stowed yaw, so only then extend slightly first. At table height the
            # arm stays retracted until the base is in position.
            low_lift_yaw_clearance_m = 0.4
            if lift_to_pregrasp_m < low_lift_yaw_clearance_m:
                self.logger.info('Extend wrist slightly to clear the base before rotating the gripper.')
                self.debug_move_to_pose({'wrist_extension': 0.12})

            self.logger.info('Rotate the gripper for grasping.')
            self.debug_move_to_pose({'joint_wrist_yaw': pregrasp_yaw})

            self.logger.info('Open the gripper.')
            self.debug_move_to_pose({'gripper_aperture': 0.07})

            pregrasp_mobile_base_m, pregrasp_wrist_extension_m = (
                manip.get_pregrasp_planar_translation(grasp_target, self.tf2_buffer)
            )

            self.logger.info(f"debug pregrasp_mobile_base: {pregrasp_mobile_base_m}\n debug pregrasp_wrist_extension: {pregrasp_wrist_extension_m}")

            # The base only drives forward/back, so an object further out than the arm
            # reaches would otherwise get a short grasp at the clamped extension.
            if self.wrist_position is not None:
                needed_extension_m = self.wrist_position + pregrasp_wrist_extension_m
                if needed_extension_m > max_extension_m:
                    message = (f'Object is out of reach: the arm needs {needed_extension_m:.2f} m of extension '
                               f'(max {max_extension_m:.2f} m), {100 * (needed_extension_m - max_extension_m):.0f} cm too far. '
                               'Move the base closer to the object.')
                    self.logger.error(message)
                    response.success = False
                    response.message = message
                    return response

            self.logger.info('Drive to pregrasp location.')
            self.debug_drive(pregrasp_mobile_base_m-0.01)

            if self.wrist_position is None:
                self.logger.error('wrist position unavailable')
                response.success = False
                response.message = 'wrist position unavailable'
                return response

            # The base is now in line with the object, so extend over it at the
            # pregrasp height.
            if pregrasp_wrist_extension_m > 0.0:
                extension_m = max(self.wrist_position + pregrasp_wrist_extension_m, min_extension_m)
                extension_m = min(extension_m, max_extension_m)
                self.logger.info('Extend tool above the object.')
                self.debug_move_to_pose({'wrist_extension': extension_m})

            grasp_mobile_base_m, grasp_lift_m, grasp_wrist_extension_m = (
                manip.get_grasp_from_pregrasp(grasp_target, self.tf2_buffer)
            )
            lift_m = max(self.lift_position + grasp_lift_m, min_lift_m)
            lift_m = min(lift_m, max_lift_m)
            extension_m = max(self.wrist_position + grasp_wrist_extension_m, min_extension_m)
            if extension_m > max_extension_m + 0.01:
                self.logger.warn(
                    f'Grasp needs {extension_m:.3f} m of extension but the arm stops at {max_extension_m:.2f} m, '
                    f'so the grasp will be {100 * (extension_m - max_extension_m):.0f} cm short.'
                )
            extension_m = min(extension_m, max_extension_m)

            # Finish the horizontal alignment above the object first, then lower
            # straight down, so the gripper never moves diagonally into the surface.
            self.logger.info('Align the gripper above the object.')
            self.debug_move_to_pose({
                'translate_mobile_base': grasp_mobile_base_m-0.01,
                'wrist_extension': extension_m
            })

            self.logger.info('Lower the gripper onto the object.')
            self.debug_move_to_pose({'joint_lift': lift_m})

            self.logger.info('Attempt to close the gripper on the object.')
            gripper_aperture_m = max(0.0, assumed_width_m - 0.18)
            self.debug_move_to_pose({'gripper_aperture': gripper_aperture_m})

            time.sleep(3.0)

            self.logger.info('Attempt to lift the object.')
            object_lift_height_m = 0.5
            lift_m = max(self.lift_position + object_lift_height_m, 0.6)
            lift_m = min(lift_m, max_lift_m)
            self.debug_move_to_pose({'joint_lift': lift_m})

            self._present_gripper_to_camera()
            self._clear_translation_marker()

            if not self.verify_grasp_visually:
                response.success = True
                response.message = 'Object grasp attempted and carried; grasp not verified (verify_grasp_visually is false).'
                return response

            response.success, response.message = self._verify_grasp(request.expected_label, pick_odom)
            if response.success:
                self.logger.info(response.message)
            else:
                self.logger.error(response.message)
            return response
        except DebugMoveCancelled as exc:
            self._clear_translation_marker()
            self.logger.warn(str(exc))
            response.success = False
            response.message = str(exc)
            return response

    def _declare_parameter_once(self, name, default):
        # HelloNode auto-declares parameters passed as overrides (--params-file, -p),
        # so only declare the ones that were not overridden.
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def main(self):
        hm.HelloNode.main(self, 'grasp_carry', 'grasp_carry', wait_for_first_pointcloud=False)
        self.logger = self.get_logger()

        self.callback_group = ReentrantCallbackGroup()

        self._declare_parameter_once('debug_directory', '')
        self._declare_parameter_once('assumed_width_m', 0.05)
        self._declare_parameter_once('scan_time_s', 2.0)
        self._declare_parameter_once('grasp_service', '/grasp_carry')
        self._declare_parameter_once('dryrun', False)
        self._declare_parameter_once('debug_mode', False)
        self._declare_parameter_once('carry_wrist_pitch_rad', -0.3)
        self._declare_parameter_once('carry_wrist_yaw_offset_rad', 0.5)
        self._declare_parameter_once('look_at_gripper', True)
        self._declare_parameter_once('stow_before_grasp', False)
        self._declare_parameter_once('verify_grasp_visually', True)

        self.debug_directory = self.get_parameter('debug_directory').value
        self.assumed_width_m = float(self.get_parameter('assumed_width_m').value)
        self.scan_time_s = float(self.get_parameter('scan_time_s').value)
        grasp_service = self.get_parameter('grasp_service').value

        self.dryrun = self.get_parameter('dryrun').value
        self.debug_mode = bool(self.get_parameter('debug_mode').value)
        self.carry_wrist_pitch_rad = float(self.get_parameter('carry_wrist_pitch_rad').value)
        self.carry_wrist_yaw_offset_rad = float(self.get_parameter('carry_wrist_yaw_offset_rad').value)
        self.look_at_gripper = bool(self.get_parameter('look_at_gripper').value)
        self.stow_before_grasp = bool(self.get_parameter('stow_before_grasp').value)
        self.verify_grasp_visually = bool(self.get_parameter('verify_grasp_visually').value)
        self._setup_grasp_check()

        self.joint_states_subscriber = self.create_subscription(
            JointState, '/stretch/joint_states', callback=self.joint_states_callback, qos_profile=1, callback_group=self.callback_group)

        self.grasp_carry_service = self.create_service(
            GraspPoint, grasp_service, self.grasp_carry_service_callback, callback_group=self.callback_group)

        self.debug_marker_pub = self.create_publisher(
            Marker, '/grasp_carry/debug_translation_marker', 1)

        self.logger.info(f'Grasp point node is ready. Service: {grasp_service}')


def main():
    try:
        node = GraspPointNode()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        rclpy.logging.get_logger('grasp_carry').info('interrupt received, so shutting down')


if __name__ == '__main__':
    main()
