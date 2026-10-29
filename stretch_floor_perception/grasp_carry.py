#!/usr/bin/env python3

from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState
from visualization_msgs.msg import Marker

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
import rclpy.logging

import json
import math
import time
import threading
import numpy as np
import os
import cv2
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_srvs.srv import Trigger

import ros2_numpy as rn
from scipy import ndimage

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
        # Wrist pose for checking the grasp with the head camera.
        self.carry_wrist_yaw_rad = 0.70
        self.carry_wrist_pitch_rad = -0.02
        self.carry_wrist_roll_rad = 2.91
        # Roll to whichever of carry_wrist_roll_rad and its half-turn opposite is nearer.
        self.carry_wrist_roll_nearest = False
        self.look_at_gripper = True
        self.stow_before_grasp = False
        self.verify_grasp_visually = True
        self.object_lift_height_m = 0.10
        # Turning to close the side grip across the object's short side.
        self.align_side_grasp = True
        self.side_min_elongation_ratio = 1.4
        self.max_base_turn_rad = 0.52
        self.max_side_wrist_yaw_rad = 0.8
        self.side_max_grip_width_m = 0.10
        # Squaring the base to the table edge found in the funmap surface scan.
        self.square_to_table_edge = True
        self.max_square_turn_rad = 0.52
        # Looking at the object again after the base has turned and driven.
        self.reobserve_after_base_move = True
        self.max_reobserve_shift_m = 0.05
        self.head_settle_s = 1.0
        # Measuring the object in the point cloud (topdown_grasp overrides these from parameters).
        self.object_cluster_radius_m = 0.12
        self.object_vertical_window_m = 0.30
        self.min_object_height_above_surface_m = 0.01
        self.max_object_height_m = 0.30
        self.min_object_points = 60
        self.min_elongation_ratio = 1.6
        self.top_band_fraction = 0.35
        self.min_top_band_m = 0.01
        self.bbox_margin_m = 0.03
        self.cluster_cell_m = 0.01
        self.min_cluster_points = 15
        # Grasp heights worked out with the measurement, used by topdown_grasp.
        self.grasp_depth_below_top_m = 0.04
        self.surface_clearance_m = 0.015
        self.hover_above_object_m = 0.20
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

        if self.debug_directory:
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
        # Move the wrist to a fixed pose for the grasp check: swung sideways (positive
        # yaw is toward the front of the robot) so the arm does not hide the object,
        # level, and rolled so the held object lies on its side facing the camera.
        pose = {'joint_wrist_yaw': float(np.clip(self.carry_wrist_yaw_rad, -1.75, 4.0))}

        if self.get_tool() == "tool_stretch_dex_wrist":
            pose['joint_wrist_pitch'] = self.carry_wrist_pitch_rad
            pose['joint_wrist_roll'] = self._carry_roll()

        if pose:
            self.logger.info('Turn the gripper to the side and roll the object so it is visible to the head camera.')
            self.debug_move_to_pose(pose)

        if self.look_at_gripper:
            self._point_head_at_gripper_tip()

    def _carry_roll(self):
        """Wrist roll for the grasp check pose.

        The two fingers are alike, so the gripper rolled half a turn holds the object the same
        way, just upside down. With carry_wrist_roll_nearest, roll to whichever of the two is
        nearer the current roll, so the held object is not twisted round by up to a full turn.
        """
        roll = float(self.carry_wrist_roll_rad)
        current = self._get_joint_position('joint_wrist_roll')
        if not self.carry_wrist_roll_nearest or current is None:
            return roll
        opposite = roll - np.pi if roll > 0.0 else roll + np.pi
        if abs(opposite - current) < abs(roll - current):
            self.logger.info(f'Roll the wrist to {opposite:.2f} rad rather than {roll:.2f} rad, '
                             f'which is nearer its current {current:.2f} rad.')
            return opposite
        return roll

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

    # ------------------------------------------------------------------
    # Shared geometry and debug markers (also used by topdown_grasp)
    # ------------------------------------------------------------------

    def _get_frame_in_base(self, frame_id):
        frame_to_base_mat, _ = hm.get_p1_to_p2_matrix(frame_id, 'base_link', self.tf2_buffer)
        if frame_to_base_mat is None:
            return None
        return frame_to_base_mat[:3, 3].copy()

    def _bbox_in_base(self, bbox, frame_id):
        """The detector's box in base_link, or None when the request has no box.

        Returns center, the unit long axis in the xy plane, and size (long, short, height).
        """
        size = np.array([bbox.size.x, bbox.size.y, bbox.size.z], dtype=float)
        if not np.all(np.isfinite(size)) or size[0] <= 0.0 or size[1] <= 0.0:
            return None
        p = bbox.center.position
        center_frame = np.array([p.x, p.y, p.z], dtype=float)
        center = self._transform_point(center_frame, frame_id, 'base_link')
        if center is None:
            return None

        # The x axis of the box orientation, carried into base_link. An unset (all zero)
        # quaternion gives the frame's x axis.
        q = bbox.center.orientation
        axis_frame = np.array([
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            2.0 * (q.x * q.y + q.w * q.z),
            2.0 * (q.x * q.z - q.w * q.y),
        ])
        axis_end = self._transform_point(center_frame + axis_frame, frame_id, 'base_link')
        axis_xy = (axis_end - center)[:2] if axis_end is not None else np.array([1.0, 0.0])
        norm = float(np.linalg.norm(axis_xy))
        axis_xy = axis_xy / norm if norm > 1e-6 else np.array([1.0, 0.0])

        # Keep x as the long side, whatever order the sizes come in.
        if size[1] > size[0]:
            size[0], size[1] = size[1], size[0]
            axis_xy = np.array([-axis_xy[1], axis_xy[0]])
        return {'center': center, 'major_axis_xy': axis_xy, 'size': size}

    def _publish_lines(self, ns, points_base, rgb, width_m, with_ends=False):
        """Line segments (pairs of base_link points) in odom, optionally with spheres at the ends."""
        if self.debug_marker_pub is None:
            return
        points_odom = [self._transform_point(p, 'base_link', 'odom') for p in points_base]
        if any(p is None for p in points_odom):
            return
        points_msg = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in points_odom]
        kinds = [(0, Marker.LINE_LIST, width_m)]
        if with_ends:
            kinds.append((1, Marker.SPHERE_LIST, 2.0 * width_m))
        for marker_id, marker_type, scale in kinds:
            marker = Marker()
            marker.header.frame_id = 'odom'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = ns
            marker.id = marker_id
            marker.type = marker_type
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.points = points_msg
            marker.scale.x = marker.scale.y = marker.scale.z = float(scale)
            marker.color.r, marker.color.g, marker.color.b = rgb
            marker.color.a = 1.0
            self.debug_marker_pub.publish(marker)

    def _delete_markers(self, ns_ids):
        """Remove debug markers, given as (namespace, id) pairs."""
        if self.debug_marker_pub is None:
            return
        for ns, marker_id in ns_ids:
            marker = Marker()
            marker.header.frame_id = 'odom'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = ns
            marker.id = marker_id
            marker.action = Marker.DELETE
            self.debug_marker_pub.publish(marker)

    # ------------------------------------------------------------------
    # Measuring the object in the point cloud (also used by topdown_grasp)
    # ------------------------------------------------------------------

    def _wait_for_fresh_point_cloud(self, timeout_s=3.0):
        start_ns = self.get_clock().now().nanoseconds
        deadline_s = time.time() + timeout_s
        while time.time() < deadline_s:
            point_cloud_msg = self.point_cloud
            if Time.from_msg(point_cloud_msg.header.stamp).nanoseconds > start_ns:
                return point_cloud_msg
            time.sleep(0.05)
        return None

    def _collect_local_points_in_base_frame(self, point_cloud_msg, target_base, radius_m=None):
        if radius_m is None:
            radius_m = self.object_cluster_radius_m
        cloud_to_base_mat, _ = hm.get_p1_to_p2_matrix(
            point_cloud_msg.header.frame_id, 'base_link', self.tf2_buffer,
            lookup_time=Time.from_msg(point_cloud_msg.header.stamp), timeout_s=0.5)
        if cloud_to_base_mat is None:
            cloud_to_base_mat, _ = hm.get_p1_to_p2_matrix(
                point_cloud_msg.header.frame_id, 'base_link', self.tf2_buffer)
        if cloud_to_base_mat is None:
            return None

        xyz = rn.point_cloud2.get_xyz_points(rn.numpify(point_cloud_msg))
        if xyz.size == 0:
            return np.zeros((0, 3))
        xyz = np.matmul(xyz, cloud_to_base_mat[:3, :3].T) + cloud_to_base_mat[:3, 3]

        delta = xyz - target_base
        in_radius = (delta[:, 0] ** 2 + delta[:, 1] ** 2) <= (radius_m ** 2)
        in_window = np.abs(delta[:, 2]) <= self.object_vertical_window_m
        return xyz[in_radius & in_window]

    def _select_object_cluster(self, points, target_xy):
        """Points of the blob nearest the target, seen from above.

        Points are binned into a grid of cluster_cell_m cells, grown by one cell to
        bridge small depth holes, and split into connected blobs. Neighbouring objects
        end up in other blobs and are dropped.
        """
        cells = np.floor(points[:, :2] / self.cluster_cell_m).astype(np.int64)
        cells -= cells.min(axis=0)
        occupied = np.zeros(tuple(cells.max(axis=0) + 1), dtype=bool)
        occupied[cells[:, 0], cells[:, 1]] = True
        grown = ndimage.binary_dilation(occupied, structure=np.ones((3, 3), dtype=bool))
        labels, num_labels = ndimage.label(grown, structure=np.ones((3, 3), dtype=bool))
        point_labels = labels[cells[:, 0], cells[:, 1]]

        best_label, best_dist = None, math.inf
        for label in range(1, num_labels + 1):
            member_xy = points[point_labels == label, :2]
            if len(member_xy) < self.min_cluster_points:
                continue
            dist = float(np.min(np.linalg.norm(member_xy - target_xy, axis=1)))
            if dist < best_dist:
                best_label, best_dist = label, dist
        if best_label is None:
            return None
        if num_labels > 1:
            self.logger.info(
                f'Kept the object blob nearest the target out of {num_labels} '
                f'({int(np.count_nonzero(point_labels == best_label))} of {len(points)} points).'
            )
        return points[point_labels == best_label]

    def _make_object(self, center_xy, surface_z, top_z, major_axis_xy, major_length_m, minor_length_m,
                     fallback_width_m, elongation_ratio=None):
        major_axis_xy = np.asarray(major_axis_xy, dtype=float)
        if elongation_ratio is None:
            elongation_ratio = major_length_m / max(minor_length_m, 1e-3)
        return {
            'center_xy': np.asarray(center_xy, dtype=float),
            'surface_z': surface_z,
            'top_z': top_z,
            'grasp_tip_z': max(top_z - self.grasp_depth_below_top_m, surface_z + self.surface_clearance_m),
            'hover_tip_z': top_z + self.hover_above_object_m,
            'major_axis_xy': major_axis_xy,
            'major_length_m': max(major_length_m, fallback_width_m),
            'minor_length_m': max(minor_length_m, fallback_width_m),
            'elongated': bool(elongation_ratio >= self.min_elongation_ratio),
            'elongation_ratio': elongation_ratio,
        }

    def _fit_footprint(self, xy):
        """Long axis, centre and (long, short) extent of points seen from above.

        The centre is the middle of the extent, not the mean of the points: the camera
        sees more of the near side, which pulls the mean towards the robot.
        """
        mean_xy = xy.mean(axis=0)
        centered_xy = xy - mean_xy
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered_xy, rowvar=False))
        major_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        major_axis /= max(np.linalg.norm(major_axis), 1e-6)
        minor_axis = np.array([-major_axis[1], major_axis[0]], dtype=float)
        lo_major, hi_major = np.percentile(centered_xy @ major_axis, [5.0, 95.0])
        lo_minor, hi_minor = np.percentile(centered_xy @ minor_axis, [5.0, 95.0])
        center_xy = (mean_xy + major_axis * 0.5 * (lo_major + hi_major)
                     + minor_axis * 0.5 * (lo_minor + hi_minor))
        return major_axis, center_xy, float(hi_major - lo_major), float(hi_minor - lo_minor)

    def _estimate_object(self, target_base, local_points_base, fallback_width_m, box=None):
        """Measure the object from the point cloud. Returns (object or None, surface_z or None)."""
        if local_points_base is None or len(local_points_base) < self.min_object_points:
            return None, None

        points = np.asarray(local_points_base, dtype=float)
        # The crop around the object is mostly the supporting surface, so a low
        # percentile of the heights is the table, desk or floor top.
        surface_z = float(np.percentile(points[:, 2], 10.0))
        object_mask = (
            (points[:, 2] > (surface_z + self.min_object_height_above_surface_m))
            & (points[:, 2] < (surface_z + self.max_object_height_m))
        )
        if box is not None:
            # Keep to the detector's box, with a margin for the side it could not see.
            delta_xy = points[:, :2] - box['center'][:2]
            major = box['major_axis_xy']
            minor = np.array([-major[1], major[0]])
            object_mask &= np.abs(delta_xy @ major) <= (0.5 * box['size'][0] + self.bbox_margin_m)
            object_mask &= np.abs(delta_xy @ minor) <= (0.5 * box['size'][1] + self.bbox_margin_m)
        object_points = points[object_mask]
        if len(object_points) < max(20, self.min_object_points // 3):
            self.logger.warn('Unable to isolate enough points above the surface for top-down grasping.')
            return None, surface_z

        target_xy = np.asarray(target_base[:2], dtype=float)
        object_points = self._select_object_cluster(object_points, target_xy)
        if object_points is None or len(object_points) < 3:
            self.logger.warn('No blob of points near the target for top-down grasping.')
            return None, surface_z

        top_z = float(np.percentile(object_points[:, 2], 95.0))
        # Fit the axis, centre and elongation to the top surface only. The side faces seen
        # at an angle and noisy edge pixels smear the footprint, so a round object can look
        # elongated in some arbitrary direction.
        band_m = max(self.top_band_fraction * (top_z - surface_z), self.min_top_band_m)
        top_points = object_points[object_points[:, 2] >= (top_z - band_m)]
        if len(top_points) < self.min_cluster_points:
            top_points = object_points
        major_axis, center_xy, top_major_m, top_minor_m = self._fit_footprint(top_points[:, :2])
        minor_axis = np.array([-major_axis[1], major_axis[0]], dtype=float)

        # Size from the whole object along those axes, so the box covers all of it.
        delta_xy = object_points[:, :2] - center_xy
        major_length_m = float(np.subtract(*np.percentile(delta_xy @ major_axis, [95.0, 5.0])))
        minor_length_m = float(np.subtract(*np.percentile(delta_xy @ minor_axis, [95.0, 5.0])))
        obj = self._make_object(center_xy, surface_z, top_z, major_axis, major_length_m, minor_length_m,
                                fallback_width_m, elongation_ratio=top_major_m / max(top_minor_m, 1e-3))
        obj['source'] = 'point cloud'
        return obj, surface_z

    def _crop_radius(self, box):
        """Radius of point cloud to take around the target: wide enough to see the surface
        around the whole detector box."""
        if box is None:
            return self.object_cluster_radius_m
        return max(self.object_cluster_radius_m,
                   0.5 * float(np.hypot(box['size'][0], box['size'][1])) + self.bbox_margin_m + 0.05)

    def _box_to_frame(self, box, source_frame, target_frame):
        """The same box with its centre and long axis carried into another frame."""
        center = self._transform_point(box['center'], source_frame, target_frame)
        axis_end = self._transform_point(box['center'] + np.append(box['major_axis_xy'], 0.0),
                                         source_frame, target_frame)
        if center is None or axis_end is None:
            return None
        axis_xy = (axis_end - center)[:2]
        axis_xy = axis_xy / max(float(np.linalg.norm(axis_xy)), 1e-6)
        return {'center': center, 'major_axis_xy': axis_xy, 'size': box['size']}

    def _reobserve_object(self, target_odom, box_odom, fallback_width_m):
        """Look at the object again after the base has moved and measure where it is now.

        Wheel odometry drifts, most of all when the base turns on the spot, so the object's
        position in odom goes stale. Returns the object measured from the fresh view (in
        base_link), or None to keep the earlier estimate.
        """
        target_base = self._transform_point(target_odom, 'odom', 'base_link')
        if target_base is None:
            return None
        self._point_head_at(target_base, 'the object again')
        time.sleep(self.head_settle_s)
        point_cloud_msg = self._wait_for_fresh_point_cloud()
        if point_cloud_msg is None:
            self.logger.warn('No fresh point cloud to look at the object again, so keeping the earlier estimate.')
            return None
        box = self._box_to_frame(box_odom, 'odom', 'base_link') if box_odom is not None else None
        points = self._collect_local_points_in_base_frame(point_cloud_msg, target_base, self._crop_radius(box))
        obj, _ = self._estimate_object(target_base, points, fallback_width_m, box)
        if obj is None:
            self.logger.warn('Could not measure the object again, so keeping the earlier estimate.')
            return None
        offset = obj['center_xy'] - target_base[:2]
        offset_m = float(np.linalg.norm(offset))
        if offset_m > self.max_reobserve_shift_m:
            self.logger.warn(
                f'Looked again, but the object measured {100 * offset_m:.1f} cm from where it was expected, '
                f'more than {100 * self.max_reobserve_shift_m:.0f} cm, so keeping the earlier estimate.')
            return None
        self.logger.info(
            f'Looked again after the base moved: the object is {100 * offset_m:.1f} cm from where it was '
            f'expected (base x {100 * offset[0]:+.1f} cm, y {100 * offset[1]:+.1f} cm), so correcting.')
        return obj

    # ------------------------------------------------------------------
    # Table edge: found in the funmap surface scan, to square the base to it
    # ------------------------------------------------------------------

    def _detect_table_edge(self, manip, target_odom):
        """The table edge in front of the object, from the funmap surface scan.

        Returns a dict in base_link (p0 on the edge, its direction, the normal pointing from
        the table towards the robot, its angle to base x and its distance from the base), or
        None if no plausible edge was found. It is drawn in RViz as a yellow line.
        """
        try:
            p0, p1, normal = manip.get_nearest_cliff('base_link', self.tf2_buffer)
        except Exception as exc:
            self.logger.warn(f'Table edge detection failed ({exc}), so assuming the robot stands square to the table.')
            return None
        if normal is None:
            self.logger.warn('No table edge found in the surface scan, so assuming the robot stands square to the table.')
            return None
        p0 = np.asarray(p0, dtype=float)[:2]
        p1 = np.asarray(p1, dtype=float)[:2]
        normal = np.asarray(normal, dtype=float)[:2]
        normal = normal / max(float(np.linalg.norm(normal)), 1e-6)
        direction = (p1 - p0) / max(float(np.linalg.norm(p1 - p0)), 1e-6)
        target_base = self._transform_point(target_odom, 'odom', 'base_link')
        if target_base is None:
            return None
        # The object should sit on the table, beyond the edge as seen from the robot.
        inside_m = float(np.dot(p0 - target_base[:2], normal))
        if inside_m < -0.02:
            self.logger.warn('The nearest edge found lies beyond the object, so it is not the table edge; ignoring it.')
            return None
        angle = float(np.arctan2(direction[1], direction[0]))
        angle = (angle + np.pi / 2.0) % np.pi - np.pi / 2.0
        distance_m = -float(np.dot(p0, normal))
        self.logger.info(
            f'Table edge found {distance_m:.2f} m from the base, at {np.degrees(angle):+.0f} deg to the robot; '
            f'the object is {100 * inside_m:.0f} cm in from it.')
        z = float(target_base[2])
        self._publish_lines('table_edge', [np.append(p0, z), np.append(p1, z)], (1.0, 0.85, 0.0), 0.008)
        return {'p0': p0, 'direction': direction, 'normal': normal, 'angle': angle, 'distance_m': distance_m}

    def _square_turn(self, edge):
        """Base turn (counterclockwise, radians) that points the arm straight at the table edge,
        or 0 when the edge is too far off square to trust."""
        normal = edge['normal']
        # Square, the arm (base -y) points at the table, so the normal towards the robot is +y.
        turn = float(np.arctan2(normal[1], normal[0])) - np.pi / 2.0
        turn = (turn + np.pi) % (2.0 * np.pi) - np.pi
        if abs(turn) > self.max_square_turn_rad:
            self.logger.warn(
                f'The table edge is {np.degrees(turn):+.0f} deg off square, more than '
                f'{np.degrees(self.max_square_turn_rad):.0f} deg, so not squaring up to it '
                '(it may not be the edge in front of the robot).')
            return 0.0
        return turn

    # ------------------------------------------------------------------
    # Side grasp alignment: turn so the fingers close along the object axis nearest the edge
    # ------------------------------------------------------------------

    def _plan_side_alignment(self, box, edge_angle=0.0, edge_source='assumed'):
        """Base turn and extra wrist yaw (both counterclockwise, radians) for the side grasp.

        The gripper points straight out along the arm (base -y), coming in from the table
        edge, and its fingers close along base x, turned by however much the base and wrist
        turn. edge_angle is the table edge's angle to base x: measured from the surface scan,
        or 0 when the robot is assumed to stand side-on to the table. The fingers close along
        whichever of the object's two axes is closest to the edge. The base takes up to max_base_turn_rad of the
        turn, the wrist yaw up to max_side_wrist_yaw_rad of the rest.
        Returns (base_turn, wrist_yaw, closing_xy), closing_xy being the planned closing
        direction in the current base_link, or None when no turn is wanted.
        """
        if (not self.align_side_grasp) or box is None:
            return 0.0, 0.0, None
        ratio = box['size'][0] / max(box['size'][1], 1e-3)
        if ratio < self.side_min_elongation_ratio:
            self.logger.info(f'Object footprint is nearly round (ratio {ratio:.2f}), so no turn for the side grasp.')
            return 0.0, 0.0, None

        def fold(angle):
            # Closing along a line works either way round, so take the smaller turn.
            return (angle + np.pi / 2.0) % np.pi - np.pi / 2.0

        major = box['major_axis_xy']
        major_angle = float(np.arctan2(major[1], major[0]))
        # (turn, side gripped across, its width, how far that closing line is from the edge)
        across_narrow = (fold(major_angle + np.pi / 2.0), 'narrow', float(box['size'][1]),
                         abs(fold(major_angle + np.pi / 2.0 - edge_angle)))
        across_wide = (fold(major_angle), 'wide', float(box['size'][0]), abs(fold(major_angle - edge_angle)))
        # Close along the axis closest to the edge; on a tie, grip the narrow side.
        turn, side, width, _ = across_narrow if across_narrow[3] <= across_wide[3] + 1e-6 else across_wide

        base_turn = float(np.clip(turn, -self.max_base_turn_rad, self.max_base_turn_rad))
        wrist_yaw = float(np.clip(turn - base_turn, -self.max_side_wrist_yaw_rad, self.max_side_wrist_yaw_rad))
        residual = turn - base_turn - wrist_yaw
        total = base_turn + wrist_yaw
        self.logger.info(
            f'Side grasp alignment: long axis at {np.degrees(major_angle):.0f} deg, ratio {ratio:.2f}; '
            f'turn base {np.degrees(base_turn):+.0f} deg and wrist {np.degrees(wrist_yaw):+.0f} deg so the '
            f'fingers close across the {side} side ({100 * width:.1f} cm), the axis closest to the table edge '
            f'({edge_source}, at {np.degrees(edge_angle):+.0f} deg).'
        )
        if abs(residual) > np.radians(2.0):
            self.logger.warn(f'Could only turn {np.degrees(total):+.0f} of {np.degrees(turn):+.0f} deg, '
                             f'so the grip will be {np.degrees(residual):+.0f} deg off.')
        if width > self.side_max_grip_width_m:
            self.logger.warn(f'The {side} side measures {100 * width:.1f} cm, wider than the gripper\'s '
                             f'{100 * self.side_max_grip_width_m:.1f} cm, so the grip may not fit '
                             '(the detector box can overstate upright objects).')
        return base_turn, wrist_yaw, np.array([np.cos(total), np.sin(total)])

    def _publish_planned_side_grip(self, box, closing_xy):
        """Show the object's long axis (blue) and where the fingers will close (green)."""
        center = box['center']
        half = 0.5 * box['size'][0] * np.append(box['major_axis_xy'], 0.0)
        self._publish_lines('side_long_axis', [center - half, center + half], (0.2, 0.4, 1.0), 0.006)
        half = 0.5 * 0.07 * np.append(closing_xy, 0.0)  # 0.07 is the opened gripper aperture
        self._publish_lines('side_grip_planned', [center - half, center + half], (0.1, 0.9, 0.1), 0.008,
                            with_ends=True)

    def _publish_actual_side_grip(self, planned_closing_odom):
        """Show where the fingertips really are (magenta) and log how far their closing line is
        from the planned one."""
        left = self._get_frame_in_base('link_gripper_fingertip_left')
        right = self._get_frame_in_base('link_gripper_fingertip_right')
        if left is None or right is None:
            return
        self._publish_lines('side_grip_actual', [left, right], (1.0, 0.1, 0.9), 0.006, with_ends=True)
        left_odom = self._transform_point(left, 'base_link', 'odom')
        right_odom = self._transform_point(right, 'base_link', 'odom')
        if left_odom is None or right_odom is None or planned_closing_odom is None:
            return
        across = (right_odom - left_odom)[:2]
        if float(np.linalg.norm(across)) < 0.01:
            self.logger.info('Fingertips too close together to measure the closing direction.')
            return
        actual_deg = np.degrees(np.arctan2(across[1], across[0]))
        planned_deg = np.degrees(np.arctan2(planned_closing_odom[1], planned_closing_odom[0]))
        # Closing along a line is the same either way round, so compare modulo 180 degrees.
        error_deg = (actual_deg - planned_deg + 90.0) % 180.0 - 90.0
        self.logger.info(f'Fingers close along {actual_deg % 180.0:.0f} deg, planned {planned_deg % 180.0:.0f} deg '
                         f'(odom), off by {error_deg:+.0f} deg.')

    def _clear_side_markers(self):
        self._delete_markers((('side_long_axis', 0), ('side_grip_planned', 0), ('side_grip_planned', 1),
                              ('side_grip_actual', 0), ('side_grip_actual', 1), ('table_edge', 0)))

    def _color_callback(self, msg):
        with self.color_lock:
            self.latest_color = msg

    def _color_info_callback(self, msg):
        with self.color_lock:
            self.latest_color_info = msg

    def _setup_grasp_check(self):
        for name, default in (
            ('carry_wrist_roll_rad', 2.91),
            ('carry_wrist_roll_nearest', False),
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
            # Trained grasp verifier (verify_grasp_server.py). It moves to the check pose its
            # model was trained in. '' uses only the built-in check below.
            ('verifier_service', '/verify_grasp'),
            ('verifier_timeout_s', 30.0),
            ('verifier_retries', 1),
        ):
            self._declare_parameter_once(name, default)
        self.carry_wrist_roll_rad = float(self.get_parameter('carry_wrist_roll_rad').value)
        self.carry_wrist_roll_nearest = bool(self.get_parameter('carry_wrist_roll_nearest').value)
        self.grasp_check_settle_s = float(self.get_parameter('grasp_check_settle_s').value)
        self.grasp_check_frames = int(self.get_parameter('grasp_check_frames').value)
        self.grasp_check_min_frames = int(self.get_parameter('grasp_check_min_frames').value)
        self.grasp_check_min_score = float(self.get_parameter('grasp_check_min_score').value)
        self.grasp_check_roi_radius_m = float(self.get_parameter('grasp_check_roi_radius_m').value)
        self.grasp_check_timeout_s = float(self.get_parameter('grasp_check_timeout_s').value)
        self.check_pick_location = bool(self.get_parameter('check_pick_location').value)
        self.pick_check_radius_m = float(self.get_parameter('pick_check_radius_m').value)
        self.accept_pick_location_clear = bool(self.get_parameter('accept_pick_location_clear').value)
        self.verifier_timeout_s = float(self.get_parameter('verifier_timeout_s').value)
        self.verifier_retries = int(self.get_parameter('verifier_retries').value)
        verifier_service = str(self.get_parameter('verifier_service').value)
        self.verifier_client = (
            self.create_client(Trigger, verifier_service, callback_group=self.callback_group)
            if verifier_service else None)

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

    @staticmethod
    def _label_key(label):
        """Label for comparing names across models: 'Chick peas', 'chick_peas' and 'chickpeas' match."""
        return ''.join(ch for ch in str(label).lower() if ch.isalnum())

    def _verifier_ready(self):
        return self.verifier_client is not None and self.verifier_client.wait_for_service(timeout_sec=0.5)

    def _call_verifier(self):
        """One /verify_grasp call. Returns its result dict, or None if the call itself failed."""
        future = self.verifier_client.call_async(Trigger.Request())
        deadline_s = time.time() + self.verifier_timeout_s
        while not future.done() and time.time() < deadline_s:
            time.sleep(0.05)
        if not future.done() or future.result() is None:
            self.logger.error(f'Grasp verifier {self.verifier_client.srv_name} did not respond in time.')
            return None
        try:
            return json.loads(future.result().message)
        except ValueError:
            self.logger.error(f'Grasp verifier gave an unreadable reply: {future.result().message!r}')
            return None

    def _check_with_verifier(self, expected):
        """Ask the trained grasp verifier what the gripper holds. Returns (status, detail) like
        _check_label_in_gripper, plus 'held' when it holds something the verifier was not
        trained to name, so it cannot say whether it is the expected item."""
        result = None
        for attempt in range(1 + max(0, self.verifier_retries)):
            result = self._call_verifier()
            if result is None or 'error' in result:
                reason = result.get('error') if result else 'no reply'
                return 'unavailable', f'grasp verifier failed ({reason})'
            self.logger.info(f'Grasp check (verifier): {json.dumps(result)}')
            if not result.get('uncertain'):
                break
            if attempt < self.verifier_retries:
                self.logger.info('Grasp verifier was uncertain, so checking again.')

        label, item = str(result.get('label', '')), str(result.get('item', ''))
        conf = float(result.get('conf', 0.0))
        votes = f"{result.get('votes', {})} over {result.get('frames', 0)} frames"
        if result.get('uncertain'):
            return 'unseen', f"grasp verifier uncertain: '{label}' ({conf:.2f}), votes {votes}"
        if not item:
            return 'unseen', f"grasp verifier sees the gripper as '{label}' ({conf:.2f})"
        if not expected or self._label_key(item) == self._label_key(expected):
            return 'confirmed', f"grasp verifier sees '{item}' ({conf:.2f}) in the gripper, votes {votes}"
        known = {self._label_key(c) for c in result.get('classes', [])}
        if known and self._label_key(expected) not in known:
            return 'held', (f"grasp verifier sees something held ('{item}', {conf:.2f}) but is not trained "
                            f"on '{expected}', so cannot confirm which item")
        return 'wrong', f"holding '{item}' ({conf:.2f}), expected '{expected}'"

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

    def _verify_grasp(self, expected_label, pick_odom, use_verifier=False):
        """Returns (success, message) for the service response.

        With use_verifier, the trained grasp verifier checks the gripper (it moves to its own
        check pose). If it fails, the gripper is shown to the camera and checked as before.
        """
        if self.dryrun:
            return True, 'dryrun, grasp check skipped'
        expected = self._normalize_label(expected_label) if expected_label else ''

        gripper_status = 'unavailable'
        if use_verifier:
            gripper_status, gripper_detail = self._check_with_verifier(expected)
            if gripper_status == 'unavailable':
                self.logger.warn(f'{gripper_detail}, so checking with the detector instead.')
                self._present_gripper_to_camera()
        if gripper_status == 'unavailable':
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
        if gripper_status == 'held':
            return True, 'Held (item not identified): ' + '; '.join(details)
        if pick_status == 'clear' and self.accept_pick_location_clear:
            return True, 'Likely held (not identified in the gripper): ' + '; '.join(details)
        return False, 'Not confirmed: ' + '; '.join(details)

    def grasp_carry_service_callback(self, request, response):
        try:
            # Drop the markers planned for the previous object.
            self._clear_side_markers()
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

            # Scan the surface and find the table edge, then turn so the fingers close along
            # the object axis nearest the edge, or, for a round object, square up to the edge.
            # base_link turns with the robot, so the target and box are followed in odom.
            if pick_odom is None:
                message = 'Unable to locate the target in odom, so unable to plan the grasp.'
                self.logger.error(message)
                response.success = False
                response.message = message
                return response
            box = self._bbox_in_base(request.bbox, msg.header.frame_id)
            box_odom = self._box_to_frame(box, 'base_link', 'odom') if box is not None else None

            manip = self.look_at_surface(scan_time_s=self.scan_time_s)
            if manip is None:
                self.logger.error('Failed to build manipulation view.')
                response.success = False
                response.message = 'Failed to build manipulation view.'
                return response

            edge = self._detect_table_edge(manip, pick_odom) if self.square_to_table_edge else None
            base_turn_rad, side_wrist_yaw_rad, closing_xy = self._plan_side_alignment(
                box, edge['angle'] if edge is not None else 0.0, 'measured' if edge is not None else 'assumed')
            if closing_xy is None and edge is not None:
                base_turn_rad = self._square_turn(edge)
            planned_closing_odom = None
            if closing_xy is not None:
                self._publish_planned_side_grip(box, closing_xy)
                origin_odom = self._transform_point(np.zeros(3), 'base_link', 'odom')
                end_odom = self._transform_point(np.append(closing_xy, 0.0), 'base_link', 'odom')
                if origin_odom is not None and end_odom is not None:
                    planned_closing_odom = (end_odom - origin_odom)[:2]
            if abs(base_turn_rad) > 1e-3:
                if closing_xy is None:
                    self.logger.info(f'Turn the base {np.degrees(base_turn_rad):+.0f} deg to square up to the table edge.')
                else:
                    self.logger.info('Turn the base to line the gripper up with the object.')
                self.debug_move_to_pose({'rotate_mobile_base': base_turn_rad})
                msg = PointStamped()
                msg.header.frame_id = 'odom'
                msg.header.stamp = request.target.header.stamp
                msg.point = Point(x=float(pick_odom[0]), y=float(pick_odom[1]), z=float(pick_odom[2]))
                # Scan again from where the grasp is made.
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
            self.debug_move_to_pose({'joint_wrist_yaw': pregrasp_yaw + side_wrist_yaw_rad})

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

            # Look again: odometry drifts while turning and driving, so measure the object from
            # where the base now is and move funmap's target onto it.
            if self.reobserve_after_base_move:
                target_odom = self._transform_point(
                    [msg.point.x, msg.point.y, msg.point.z], msg.header.frame_id, 'odom')
                again = (self._reobserve_object(target_odom, box_odom, assumed_width_m)
                         if target_odom is not None else None)
                target_base = (self._transform_point(target_odom, 'odom', 'base_link')
                               if again is not None else None)
                corrected_odom = None
                if target_base is not None:
                    corrected_odom = self._transform_point(
                        np.array([again['center_xy'][0], again['center_xy'][1], target_base[2]]), 'base_link', 'odom')
                if corrected_odom is not None:
                    odom_to_image_mat, _ = max_height_im.get_points_to_image_mat('odom', self.tf2_buffer)
                    if odom_to_image_mat is not None:
                        corrected_pix = np.matmul(odom_to_image_mat, np.append(corrected_odom, 1.0))[:3]
                        grasp_target['location_xy_pix'] = np.array(corrected_pix[:2])
                        grasp_target['location_z_pix'] = corrected_pix[2]

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
            # Fingers are open and over the object: show and check their real closing line.
            self._publish_actual_side_grip(planned_closing_odom)

            self.logger.info('Lower the gripper onto the object.')
            self.debug_move_to_pose({'joint_lift': lift_m})

            self.logger.info('Attempt to close the gripper on the object.')
            gripper_aperture_m = max(0.0, assumed_width_m - 0.18)
            self.debug_move_to_pose({'gripper_aperture': gripper_aperture_m})

            time.sleep(3.0)

            # Lift just clear of the surface for the grasp check; the carry pose comes afterwards.
            self.logger.info('Attempt to lift the object.')
            lift_m = min(self.lift_position + self.object_lift_height_m, max_lift_m)
            self.debug_move_to_pose({'joint_lift': lift_m})

            # The grasp verifier moves to its own check pose; otherwise show the gripper here.
            use_verifier = self.verify_grasp_visually and (not self.dryrun) and self._verifier_ready()
            if not use_verifier:
                self._present_gripper_to_camera()
            self._clear_translation_marker()
            self._clear_side_markers()

            if not self.verify_grasp_visually:
                response.success = True
                response.message = 'Object grasp attempted and carried; grasp not verified (verify_grasp_visually is false).'
                return response

            response.success, response.message = self._verify_grasp(
                request.expected_label, pick_odom, use_verifier)
            if response.success:
                self.logger.info(response.message)
            else:
                self.logger.error(response.message)
            return response
        except DebugMoveCancelled as exc:
            self._clear_translation_marker()
            self._clear_side_markers()
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
        self._declare_parameter_once('carry_wrist_pitch_rad', -0.02)
        self._declare_parameter_once('carry_wrist_yaw_rad', 0.70)
        self._declare_parameter_once('look_at_gripper', True)
        self._declare_parameter_once('stow_before_grasp', False)
        self._declare_parameter_once('verify_grasp_visually', True)
        # How far to lift the object off the surface before checking the grasp.
        self._declare_parameter_once('object_lift_height_m', 0.10)
        self._declare_parameter_once('align_side_grasp', True)
        self._declare_parameter_once('side_min_elongation_ratio', 1.4)
        # The base turns first (it costs no reach), then the wrist takes the rest of the turn,
        # which is at most 45 deg.
        self._declare_parameter_once('max_base_turn_rad', 0.52)
        self._declare_parameter_once('max_side_wrist_yaw_rad', 0.8)
        # Widest side the open gripper is expected to fit around; only used for a warning.
        self._declare_parameter_once('side_max_grip_width_m', 0.10)
        # Square up to (or line up with) the table edge found in the surface scan.
        self._declare_parameter_once('square_to_table_edge', True)
        self._declare_parameter_once('max_square_turn_rad', 0.52)
        # Look at the object again after the base moves, and correct by up to this much.
        self._declare_parameter_once('reobserve_after_base_move', True)
        self._declare_parameter_once('max_reobserve_shift_m', 0.05)

        self.debug_directory = self.get_parameter('debug_directory').value
        self.assumed_width_m = float(self.get_parameter('assumed_width_m').value)
        self.scan_time_s = float(self.get_parameter('scan_time_s').value)
        grasp_service = self.get_parameter('grasp_service').value

        self.dryrun = self.get_parameter('dryrun').value
        self.debug_mode = bool(self.get_parameter('debug_mode').value)
        self.carry_wrist_pitch_rad = float(self.get_parameter('carry_wrist_pitch_rad').value)
        self.carry_wrist_yaw_rad = float(self.get_parameter('carry_wrist_yaw_rad').value)
        self.look_at_gripper = bool(self.get_parameter('look_at_gripper').value)
        self.stow_before_grasp = bool(self.get_parameter('stow_before_grasp').value)
        self.verify_grasp_visually = bool(self.get_parameter('verify_grasp_visually').value)
        self.object_lift_height_m = float(self.get_parameter('object_lift_height_m').value)
        self.align_side_grasp = bool(self.get_parameter('align_side_grasp').value)
        self.side_min_elongation_ratio = float(self.get_parameter('side_min_elongation_ratio').value)
        self.max_base_turn_rad = float(self.get_parameter('max_base_turn_rad').value)
        self.max_side_wrist_yaw_rad = float(self.get_parameter('max_side_wrist_yaw_rad').value)
        self.side_max_grip_width_m = float(self.get_parameter('side_max_grip_width_m').value)
        self.square_to_table_edge = bool(self.get_parameter('square_to_table_edge').value)
        self.max_square_turn_rad = float(self.get_parameter('max_square_turn_rad').value)
        self.reobserve_after_base_move = bool(self.get_parameter('reobserve_after_base_move').value)
        self.max_reobserve_shift_m = float(self.get_parameter('max_reobserve_shift_m').value)
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
