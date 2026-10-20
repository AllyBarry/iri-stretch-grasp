#!/usr/bin/env python3

import math
import time

import numpy as np
import rclpy
import rclpy.logging
import ros2_numpy as rn
from geometry_msgs.msg import Point, Quaternion
from rclpy.callback_groups import ReentrantCallbackGroup
from scipy import ndimage
from rclpy.time import Time
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

import hello_helpers.hello_misc as hm
from stretch_floor_perception.grasp_carry import DebugMoveCancelled, GraspNode
from stretch_floor_perception.srv import Grasp


class TopdownGraspNode(GraspNode):
    """Grasp an object from above off a table, desk or the floor, then carry it.

    The support surface height is measured from the point cloud around the target,
    so the same sequence works for surfaces at different heights. The retracted arm
    is raised above the object first and only then extends, with the gripper level.
    Once over the object the gripper tilts down, opens and descends around it. It
    ends holding the object with the arm retracted and the gripper presented to the
    head camera for visual confirmation, like grasp_carry.

    When the request carries the detector's 3D box, only points inside it (plus a
    margin) are used, and of those only the blob nearest the target, so neighbouring
    objects don't pull the grasp off centre. The fingers are aligned with the
    object's long axis. If the measured centre strays too far from the box, the box
    is used instead.
    """

    def __init__(self):
        super().__init__()
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
        self.max_center_shift_fraction = 0.5
        self.min_max_center_shift_m = 0.03
        self.roll_axis_offset_rad = 0.0
        self.topdown_wrist_pitch_rad = (-math.pi / 2.0) + 0.07
        self.topdown_wrist_yaw_rad = 0.0
        self.hover_above_object_m = 0.20
        self.min_pitch_clearance_m = 0.05
        self.align_tolerance_m = 0.005
        self.grasp_depth_below_top_m = 0.04
        self.surface_clearance_m = 0.015
        self.open_aperture_m = 0.075
        self.object_lift_height_m = 0.10
        self.retract_after_lift = True
        self.head_settle_s = 1.0

    # ------------------------------------------------------------------
    # Perception helpers
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

    def _object_from_box(self, box, surface_z, fallback_width_m):
        """The object as the detector's box describes it, on the measured surface if known."""
        center = box['center']
        size = box['size']
        top_z = float(center[2] + 0.5 * size[2])
        if surface_z is None:
            surface_z = float(center[2] - 0.5 * size[2])
        obj = self._make_object(center[:2], surface_z, top_z, box['major_axis_xy'],
                                float(size[0]), float(size[1]), fallback_width_m)
        obj['source'] = 'detector box'
        return obj

    def _grasp_axis(self, obj, box):
        """Long axis for the fingers to line up with, and where it came from (None if round)."""
        if obj['elongated']:
            return obj['major_axis_xy'], obj['source']
        if box is not None and box['size'][0] / max(box['size'][1], 1e-3) >= self.min_elongation_ratio:
            return box['major_axis_xy'], 'detector box'
        return None, 'round object'

    def _publish_object_marker(self, obj):
        """Show the planned grasp box in RViz, in odom so it stays put as the base moves."""
        if self.debug_marker_pub is None:
            return
        center_base = np.array([obj['center_xy'][0], obj['center_xy'][1],
                                0.5 * (obj['top_z'] + obj['surface_z'])])
        axis_base = np.append(obj['major_axis_xy'], 0.0)
        center_odom = self._transform_point(center_base, 'base_link', 'odom')
        axis_end_odom = self._transform_point(center_base + axis_base, 'base_link', 'odom')
        if center_odom is None or axis_end_odom is None:
            return
        yaw = math.atan2(axis_end_odom[1] - center_odom[1], axis_end_odom[0] - center_odom[0])

        marker = Marker()
        marker.header.frame_id = 'odom'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'topdown_object'
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position = Point(x=float(center_odom[0]), y=float(center_odom[1]), z=float(center_odom[2]))
        marker.pose.orientation = Quaternion(x=0.0, y=0.0, z=math.sin(0.5 * yaw), w=math.cos(0.5 * yaw))
        marker.scale.x = float(obj['major_length_m'])
        marker.scale.y = float(obj['minor_length_m'])
        marker.scale.z = float(max(obj['top_z'] - obj['surface_z'], 0.01))
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = 1.0, 0.5, 0.0, 0.5
        self.debug_marker_pub.publish(marker)

    def _planned_closing_axis(self, roll):
        """Direction, in base_link xy, the fingers are meant to close along at this wrist roll.

        _compute_topdown_roll turns the fingers to close across the long axis, with
        roll_axis_offset_rad calibrating that, so this undoes it.
        """
        angle = float(roll) - float(self.roll_axis_offset_rad) + (math.pi / 2.0)
        return np.array([math.cos(angle), math.sin(angle)])

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

    def _publish_planned_grip(self, obj, grasp_axis, roll):
        """Show the object's long axis (blue) and where the fingers will close (green, with
        spheres at the finger contacts), just above the object's top."""
        center = np.array([obj['center_xy'][0], obj['center_xy'][1], obj['top_z'] + 0.01])
        if grasp_axis is not None:
            half = 0.5 * obj['major_length_m'] * np.append(grasp_axis, 0.0)
            self._publish_lines('topdown_long_axis', [center - half, center + half], (0.2, 0.4, 1.0), 0.006)
        closing = np.append(self._planned_closing_axis(roll), 0.0)
        half = 0.5 * float(self.open_aperture_m) * closing
        self._publish_lines('topdown_grip_planned', [center - half, center + half], (0.1, 0.9, 0.1), 0.008,
                            with_ends=True)

    def _publish_actual_grip(self, roll):
        """Show where the fingertips really are (magenta) and log how far their closing line is
        from the planned one, which is the correction for roll_axis_offset_rad."""
        left = self._get_frame_in_base('link_gripper_fingertip_left')
        right = self._get_frame_in_base('link_gripper_fingertip_right')
        if left is None or right is None:
            return
        self._publish_lines('topdown_grip_actual', [left, right], (1.0, 0.1, 0.9), 0.006, with_ends=True)
        across = (right - left)[:2]
        if float(np.linalg.norm(across)) < 0.01:
            self.logger.info('Fingertips too close together to measure the closing direction.')
            return
        planned = self._planned_closing_axis(roll)
        actual_deg = math.degrees(math.atan2(across[1], across[0]))
        planned_deg = math.degrees(math.atan2(planned[1], planned[0]))
        # Closing along a line is the same either way round, so compare modulo 180 degrees.
        error_deg = (actual_deg - planned_deg + 90.0) % 180.0 - 90.0
        self.logger.info(
            f'Fingers close along {actual_deg % 180.0:.0f} deg, planned {planned_deg % 180.0:.0f} deg '
            f'(base_link), off by {error_deg:+.0f} deg. If this stays off, subtract '
            f'{math.radians(error_deg):+.3f} rad from roll_axis_offset_rad.'
        )

    def _clear_topdown_markers(self):
        """Remove the planned object and grip markers. Kept apart from _clear_translation_marker,
        which the debug prompt calls before every move."""
        if self.debug_marker_pub is None:
            return
        for ns, marker_id in (('topdown_object', 0), ('topdown_long_axis', 0),
                              ('topdown_grip_planned', 0), ('topdown_grip_planned', 1),
                              ('topdown_grip_actual', 0), ('topdown_grip_actual', 1)):
            marker = Marker()
            marker.header.frame_id = 'odom'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = ns
            marker.id = marker_id
            marker.action = Marker.DELETE
            self.debug_marker_pub.publish(marker)

    def _compute_topdown_roll(self, major_axis_xy):
        axis_xy = np.asarray(major_axis_xy, dtype=float)
        roll = math.atan2(float(axis_xy[1]), float(axis_xy[0])) + float(self.roll_axis_offset_rad)
        roll = math.atan2(math.sin(roll), math.cos(roll))

        # Top-down finger alignment is symmetric under pi rotation, so choose
        # the nearest equivalent angle within the typical wrist roll range.
        if roll > (math.pi / 2.0):
            roll -= math.pi
        elif roll < (-math.pi / 2.0):
            roll += math.pi
        return roll

    def _get_frame_in_base(self, frame_id):
        frame_to_base_mat, _ = hm.get_p1_to_p2_matrix(frame_id, 'base_link', self.tf2_buffer)
        if frame_to_base_mat is None:
            return None
        return frame_to_base_mat[:3, 3].copy()

    def _get_gripper_tip_in_base(self):
        return self._get_frame_in_base('link_grasp_center')

    def _lift_for_frame_height(self, frame_id, desired_z):
        frame_base = self._get_frame_in_base(frame_id)
        if frame_base is None or self.lift_position is None:
            return None
        return float(self.lift_position) + float(desired_z - frame_base[2])

    def _lift_for_tip_height(self, desired_tip_z):
        return self._lift_for_frame_height('link_grasp_center', desired_tip_z)

    # ------------------------------------------------------------------
    # Service
    # ------------------------------------------------------------------

    def _fail(self, response, message):
        self.logger.error(message)
        self._clear_translation_marker()
        self._clear_topdown_markers()
        response.success = False
        response.message = message
        return response

    def topdown_grasp_point_service_callback(self, request, response):
        try:
            # Drop the box planned for the previous object.
            self._clear_translation_marker()
            self._clear_topdown_markers()
            if self.get_tool() != 'tool_stretch_dex_wrist':
                return self._fail(response, 'Top-down grasping requires tool_stretch_dex_wrist.')

            msg = request.target
            assumed_width_m = (
                float(request.assumed_width_m) if request.assumed_width_m > 0.0 else float(self.assumed_width_m)
            )

            if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
                return self._fail(response, 'Target point timestamp is unset, so unable to grasp.')
            if self.lift_position is None or self.wrist_position is None:
                return self._fail(response, 'Joint state information is unavailable.')

            self.logger.info(
                'Top-down grasp point received: '
                f'frame={msg.header.frame_id} x={msg.point.x:.3f} y={msg.point.y:.3f} z={msg.point.z:.3f} '
                f"expected_label='{request.expected_label}'"
            )

            max_lift_m = 1.09
            min_lift_m = 0.07
            min_extension_m = 0.01
            max_extension_m = 0.5

            target_base = self._transform_point(
                [msg.point.x, msg.point.y, msg.point.z], msg.header.frame_id, 'base_link')
            if target_base is None:
                return self._fail(response, 'Unable to transform the target point into base_link.')
            # Remember where the object was, in odom, to check afterwards that it is gone.
            pick_odom = self._transform_point(
                [msg.point.x, msg.point.y, msg.point.z], msg.header.frame_id, 'odom')

            # 1. Retract the arm at its current height, then look at the target.
            self.logger.info('Retract the arm before planning.')
            self.debug_move_to_pose({'wrist_extension': min_extension_m})

            self._point_head_at(target_base, 'the target')
            time.sleep(self.head_settle_s)
            point_cloud_msg = self._wait_for_fresh_point_cloud()
            if point_cloud_msg is None:
                return self._fail(response, 'No fresh point cloud available, so unable to plan the top-down grasp.')

            box = self._bbox_in_base(request.bbox, msg.header.frame_id)
            crop_radius_m = self.object_cluster_radius_m
            if box is None:
                self.logger.warn('No detector box in the request, so measuring the object from the target alone.')
            else:
                self.logger.info(
                    f'Detector box: center=({box["center"][0]:.3f}, {box["center"][1]:.3f}) '
                    f'size=({box["size"][0]:.3f}, {box["size"][1]:.3f}, {box["size"][2]:.3f}) m '
                    f'axis={math.degrees(math.atan2(box["major_axis_xy"][1], box["major_axis_xy"][0])):.0f} deg'
                )
                # Wide enough to see the surface around the whole box.
                crop_radius_m = max(crop_radius_m, 0.5 * math.hypot(box['size'][0], box['size'][1])
                                    + self.bbox_margin_m + 0.05)

            local_points_base = self._collect_local_points_in_base_frame(
                point_cloud_msg, target_base, crop_radius_m)
            if local_points_base is None:
                return self._fail(response, 'Unable to transform the point cloud into base_link.')
            self.logger.info(f'Found {len(local_points_base)} point-cloud samples around the target.')

            obj, surface_z = self._estimate_object(target_base, local_points_base, assumed_width_m, box)
            if box is not None:
                # Trust the measurement only while it stays close to the detector's box.
                max_shift_m = max(self.max_center_shift_fraction * box['size'][0], self.min_max_center_shift_m)
                if obj is None:
                    self.logger.warn('Could not measure the object, so using the detector box.')
                    obj = self._object_from_box(box, surface_z, assumed_width_m)
                else:
                    shift_m = float(np.linalg.norm(obj['center_xy'] - box['center'][:2]))
                    self.logger.info(f'Measured centre is {shift_m:.3f} m from the detector box centre.')
                    if shift_m > max_shift_m:
                        self.logger.warn(
                            f'That is more than {max_shift_m:.3f} m, so using the detector box instead.')
                        obj = self._object_from_box(box, surface_z, assumed_width_m)
            elif obj is None:
                return self._fail(response, 'Unable to estimate the object and surface for top-down grasping.')
            else:
                shift_m = float(np.linalg.norm(obj['center_xy'] - target_base[:2]))
                self.logger.info(f'Measured centre is {shift_m:.3f} m from the requested target.')

            # Line the fingers up with the object's long axis; a round object needs no turn.
            grasp_axis, axis_source = self._grasp_axis(obj, box)
            topdown_roll = self._compute_topdown_roll(grasp_axis) if grasp_axis is not None else 0.0
            self._publish_object_marker(obj)
            self._publish_planned_grip(obj, grasp_axis, topdown_roll)
            self.logger.info(
                f'Estimated top-down grasp from the {obj["source"]}: '
                f'center=({obj["center_xy"][0]:.3f}, {obj["center_xy"][1]:.3f}) m '
                f'surface_z={obj["surface_z"]:.3f} m top_z={obj["top_z"]:.3f} m '
                f'grasp_tip_z={obj["grasp_tip_z"]:.3f} m hover_tip_z={obj["hover_tip_z"]:.3f} m '
                f'elongated={obj["elongated"]} ratio={obj["elongation_ratio"]:.2f} '
                f'major={obj["major_length_m"]:.3f} m minor={obj["minor_length_m"]:.3f} m '
                f'roll={topdown_roll:.3f} rad (axis from {axis_source})'
            )

            # Keep the object location in odom so it stays correct after the base moves.
            object_base = np.array([obj['center_xy'][0], obj['center_xy'][1], obj['top_z']], dtype=float)
            object_odom = self._transform_point(object_base, 'base_link', 'odom')
            if object_odom is None:
                return self._fail(response, 'Unable to transform the object location into odom.')

            def object_in_base():
                return self._transform_point(object_odom, 'odom', 'base_link')

            # Distance from the wrist pitch axis to the gripper tip. Tilting the
            # gripper down drops the tip by about this much below the pitch axis.
            tip_base = self._get_gripper_tip_in_base()
            pitch_base = self._get_frame_in_base('link_wrist_pitch')
            if tip_base is None or pitch_base is None:
                return self._fail(response, 'Unable to locate the wrist and gripper tip.')
            tip_drop_m = float(np.linalg.norm(tip_base - pitch_base)) * math.cos(
                float(self.topdown_wrist_pitch_rad) + (math.pi / 2.0))

            # 2. Raise the retracted arm so that, once tilted down over the object,
            #    the gripper tip will hover hover_above_object_m above its top. The
            #    arm only moves along the mast, so it cannot sweep through the table.
            hover_pitch_z = obj['hover_tip_z'] + tip_drop_m
            hover_lift_m = self._lift_for_frame_height('link_wrist_pitch', hover_pitch_z)
            if hover_lift_m is None:
                return self._fail(response, 'Unable to locate the wrist.')
            if hover_lift_m > max_lift_m:
                self.logger.warn(
                    f'Hover height needs lift {hover_lift_m:.3f} m, above the max {max_lift_m:.2f} m, '
                    'so hovering lower.'
                )
            hover_lift_m = min(max(hover_lift_m, min_lift_m), max_lift_m)
            self.logger.info('Raise the retracted arm above the object.')
            self.debug_move_to_pose({'joint_lift': hover_lift_m})

            pitch_base = self._get_frame_in_base('link_wrist_pitch')
            if pitch_base is None:
                return self._fail(response, 'Unable to locate the wrist.')
            predicted_tip_clearance_m = float(pitch_base[2]) - tip_drop_m - obj['top_z']
            if predicted_tip_clearance_m < self.min_pitch_clearance_m:
                tilted_tip_z = float(pitch_base[2]) - tip_drop_m
                return self._fail(
                    response,
                    f'Object too high to grasp from above: surface at {obj["surface_z"]:.3f} m, '
                    f'object top at {obj["top_z"]:.3f} m ({obj["top_z"] - obj["surface_z"]:.3f} m tall). '
                    f'At lift {float(self.lift_position):.3f} m the tilted gripper tip reaches {tilted_tip_z:.3f} m, '
                    f'so the highest object top it can clear is '
                    f'{tilted_tip_z - self.min_pitch_clearance_m:.3f} m.'
                )

            # Near the floor the gripper can hit the base while turning, so only
            # then extend slightly first.
            low_lift_turn_clearance_m = 0.4
            if hover_lift_m < low_lift_turn_clearance_m:
                self.logger.info('Extend wrist slightly to clear the base before turning the gripper.')
                self.debug_move_to_pose({'wrist_extension': 0.12})

            # 3. Point the gripper level and straight out along the arm, ready to extend.
            self.logger.info('Level the gripper, pointing out along the arm.')
            self.debug_move_to_pose({
                'joint_wrist_pitch': 0.0,
                'joint_wrist_yaw': float(self.topdown_wrist_yaw_rad),
                'joint_wrist_roll': 0.0,
            })

            # 4. Drive the base in line with the object, then extend from above so
            #    the wrist pitch axis is over the object. The arm does not extend
            #    until the base is in place.
            target_base = object_in_base()
            tip_base = self._get_gripper_tip_in_base()
            if target_base is None or tip_base is None:
                return self._fail(response, 'Unable to locate the object or the gripper tip.')
            base_delta_m = float(target_base[0] - tip_base[0])
            self.logger.info('Drive in line with the object.')
            self.debug_drive(base_delta_m)

            target_base = object_in_base()
            pitch_base = self._get_frame_in_base('link_wrist_pitch')
            if target_base is None or pitch_base is None or self.wrist_position is None:
                return self._fail(response, 'Unable to locate the object or the wrist.')
            extension_m = float(self.wrist_position) + float(pitch_base[1] - target_base[1])
            if extension_m > max_extension_m:
                return self._fail(
                    response,
                    f'Object is out of reach (needs arm extension {extension_m:.3f} m, max {max_extension_m:.2f} m).'
                )
            extension_m = max(extension_m, min_extension_m)
            self.logger.info('Extend over the object from above.')
            self.debug_move_to_pose({'wrist_extension': extension_m})

            # 5. Over the object: tilt the gripper down and align the fingers.
            self.logger.info('Tilt the gripper down over the object.')
            self.debug_move_to_pose({
                'joint_wrist_pitch': float(self.topdown_wrist_pitch_rad),
                'joint_wrist_roll': topdown_roll,
            })

            # Correct any small offset between the tilted gripper tip and the object.
            target_base = object_in_base()
            tip_base = self._get_gripper_tip_in_base()
            if target_base is None or tip_base is None or self.wrist_position is None:
                return self._fail(response, 'Unable to locate the object or the gripper tip.')
            correction_base_m = float(target_base[0] - tip_base[0])
            correction_extension_m = float(tip_base[1] - target_base[1])
            if max(abs(correction_base_m), abs(correction_extension_m)) > self.align_tolerance_m:
                extension_m = float(self.wrist_position) + correction_extension_m
                if extension_m > max_extension_m:
                    return self._fail(
                        response,
                        f'Object is out of reach (needs arm extension {extension_m:.3f} m, max {max_extension_m:.2f} m).'
                    )
                self.logger.info(
                    f'Center the gripper over the object (base {correction_base_m:+.3f} m, '
                    f'arm {correction_extension_m:+.3f} m).'
                )
                self.debug_move_to_pose({
                    'translate_mobile_base': correction_base_m,
                    'wrist_extension': max(extension_m, min_extension_m),
                })

            self.logger.info('Open the gripper.')
            self.debug_move_to_pose({'gripper_aperture': float(self.open_aperture_m)})
            # Fingers are open and over the object: show and check their real closing line.
            self._publish_actual_grip(topdown_roll)

            # 6. Continue straight down to surround the object, close and lift straight up.
            grasp_lift_m = self._lift_for_tip_height(obj['grasp_tip_z'])
            if grasp_lift_m is None:
                return self._fail(response, 'Unable to compute the grasp height.')
            grasp_lift_m = min(max(grasp_lift_m, min_lift_m), max_lift_m)
            self.logger.info('Descend around the object.')
            self.debug_move_to_pose({'joint_lift': grasp_lift_m})

            self.logger.info('Close the gripper on the object.')
            self.debug_move_to_pose({'gripper_aperture': 0.0})
            time.sleep(2.0)

            carry_lift_m = min(float(self.lift_position) + self.object_lift_height_m, max_lift_m)
            self.logger.info('Lift the object straight up off the surface.')
            self.debug_move_to_pose({'joint_lift': carry_lift_m})

            if self.retract_after_lift:
                self.logger.info('Retract the arm to carry the object.')
                self.debug_move_to_pose({'wrist_extension': min_extension_m})

            # 7. Present the gripper to the camera and confirm the object is held.
            self._present_gripper_to_camera()
            self._clear_translation_marker()
            self._clear_topdown_markers()

            if not self.verify_grasp_visually:
                response.success = True
                response.message = (
                    'Object grasp from above attempted and carried; '
                    'grasp not verified (verify_grasp_visually is false).'
                )
                return response

            response.success, response.message = self._verify_grasp(request.expected_label, pick_odom)
            if response.success:
                self.logger.info(response.message)
            else:
                self.logger.error(response.message)
            return response
        except DebugMoveCancelled as exc:
            self._clear_translation_marker()
            self._clear_topdown_markers()
            self.logger.warn(str(exc))
            response.success = False
            response.message = str(exc)
            return response

    def main(self):
        hm.HelloNode.main(self, 'topdown_grasp_point', 'topdown_grasp_point', wait_for_first_pointcloud=False)
        self.logger = self.get_logger()

        self.callback_group = ReentrantCallbackGroup()

        defaults = {
            'debug_directory': '',
            'assumed_width_m': 0.05,
            'grasp_service': '/topdown_grasp_point',
            'dryrun': False,
            'debug_mode': False,
            # Carry pose and visual confirmation (shared with grasp_carry).
            'carry_wrist_pitch_rad': -0.3,
            'carry_wrist_yaw_offset_rad': 0.5,
            'look_at_gripper': True,
            'verify_grasp_visually': True,
            # Top-down planning.
            'object_cluster_radius_m': 0.12,
            'object_vertical_window_m': 0.30,
            'min_object_height_above_surface_m': 0.01,
            'max_object_height_m': 0.30,
            'min_object_points': 60,
            'min_elongation_ratio': 1.6,
            # Height band below the top whose points give the object's axis and centre.
            'top_band_fraction': 0.35,
            'min_top_band_m': 0.01,
            # Keeping to the detected object.
            'bbox_margin_m': 0.03,
            'cluster_cell_m': 0.01,
            'min_cluster_points': 15,
            'max_center_shift_fraction': 0.5,
            'min_max_center_shift_m': 0.03,
            'roll_axis_offset_rad': 0.0,
            'topdown_wrist_pitch_rad': (-math.pi / 2.0) + 0.07,
            'topdown_wrist_yaw_rad': 0.0,
            'hover_above_object_m': 0.20,
            'min_pitch_clearance_m': 0.05,
            'align_tolerance_m': 0.005,
            'grasp_depth_below_top_m': 0.04,
            'surface_clearance_m': 0.015,
            'open_aperture_m': 0.075,
            'object_lift_height_m': 0.10,
            'retract_after_lift': True,
            'head_settle_s': 1.0,
        }
        for name, default in defaults.items():
            self._declare_parameter_once(name, default)

        grasp_service = self.get_parameter('grasp_service').value
        self.debug_directory = self.get_parameter('debug_directory').value
        self.dryrun = self.get_parameter('dryrun').value
        self.debug_mode = bool(self.get_parameter('debug_mode').value)
        self.look_at_gripper = bool(self.get_parameter('look_at_gripper').value)
        self.verify_grasp_visually = bool(self.get_parameter('verify_grasp_visually').value)
        self.retract_after_lift = bool(self.get_parameter('retract_after_lift').value)
        self.min_object_points = int(self.get_parameter('min_object_points').value)
        self.min_cluster_points = int(self.get_parameter('min_cluster_points').value)
        for name in (
            'assumed_width_m', 'carry_wrist_pitch_rad', 'carry_wrist_yaw_offset_rad',
            'object_cluster_radius_m', 'object_vertical_window_m', 'min_object_height_above_surface_m',
            'max_object_height_m', 'min_elongation_ratio', 'roll_axis_offset_rad',
            'bbox_margin_m', 'cluster_cell_m', 'max_center_shift_fraction', 'min_max_center_shift_m',
            'top_band_fraction', 'min_top_band_m',
            'topdown_wrist_pitch_rad', 'topdown_wrist_yaw_rad', 'hover_above_object_m',
            'min_pitch_clearance_m', 'align_tolerance_m', 'grasp_depth_below_top_m', 'surface_clearance_m',
            'open_aperture_m', 'object_lift_height_m', 'head_settle_s',
        ):
            setattr(self, name, float(self.get_parameter(name).value))

        # Wrist roll, camera topics and detector service for the grasp check (shared with grasp_carry).
        self._setup_grasp_check()

        self.joint_states_subscriber = self.create_subscription(
            JointState, '/stretch/joint_states', callback=self.joint_states_callback,
            qos_profile=1, callback_group=self.callback_group)

        self.grasp_point_service = self.create_service(
            Grasp, grasp_service, self.topdown_grasp_point_service_callback,
            callback_group=self.callback_group)

        self.debug_marker_pub = self.create_publisher(
            Marker, '/topdown_grasp_point/debug_translation_marker', 1)

        self.logger.info(f'Top-down grasp point node is ready. Service: {grasp_service}')


def main():
    try:
        node = TopdownGraspNode()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        rclpy.logging.get_logger('topdown_grasp_point').info('interrupt received, so shutting down')


if __name__ == '__main__':
    main()
