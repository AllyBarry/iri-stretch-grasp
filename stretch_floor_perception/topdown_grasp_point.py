#!/usr/bin/env python3

import math
import time

import numpy as np
import rclpy
import rclpy.logging
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker

import hello_helpers.hello_misc as hm
from stretch_floor_perception.grasp_point import DebugMoveCancelled, GraspPointNode
from stretch_floor_perception.srv import GraspPoint


class TopdownGraspPointNode(GraspPointNode):

    def __init__(self):
        super().__init__()
        self.object_cluster_radius_m = 0.12
        self.object_vertical_window_m = 0.20
        self.min_object_height_above_floor_m = 0.01
        self.max_object_height_m = 0.25
        self.min_object_points = 60
        self.prealign_lift_m = 0.85
        self.initial_wrist_extension_m = 0.12
        self.topdown_wrist_pitch_rad = (-math.pi / 2.0) + 0.07
        self.topdown_wrist_yaw_rad = 0.0
        self.topdown_hover_above_object_m = 0.35
        self.topdown_floor_clearance_m = 0.025
        self.topdown_below_top_m = 0.01
        self.object_lift_height_m = 0.10
        self.roll_axis_offset_rad = 0.0
        self.min_elongation_ratio = 1.35

    def _transform_points(self, points_xyz, source_frame, target_frame):
        points_xyz = np.asarray(points_xyz, dtype=float)
        if points_xyz.ndim == 1:
            points_xyz = points_xyz.reshape(1, 3)

        if source_frame == target_frame:
            return points_xyz

        transform_mat, _ = hm.get_p1_to_p2_matrix(source_frame, target_frame, self.tf2_buffer)
        if transform_mat is None:
            return None

        homogeneous = np.concatenate(
            [points_xyz, np.ones((points_xyz.shape[0], 1), dtype=float)],
            axis=1,
        )
        transformed = (transform_mat @ homogeneous.T).T
        return transformed[:, :3]

    def _transform_point(self, point_xyz, source_frame, target_frame):
        transformed = self._transform_points(point_xyz, source_frame, target_frame)
        if transformed is None or transformed.shape[0] == 0:
            return None
        return transformed[0]

    def _get_gripper_tip_in_frame(self, target_frame):
        tip_to_target_mat, _ = hm.get_p1_to_p2_matrix('link_grasp_center', target_frame, self.tf2_buffer)
        if tip_to_target_mat is None:
            self.logger.warn(f'Unable to compute the gripper tip pose in {target_frame}.')
            return None
        return tip_to_target_mat[:3, 3].copy()

    def _collect_local_points_in_base_frame(self, target_msg):
        if self.point_cloud is None:
            self.logger.error('No point cloud available yet, so unable to plan top-down grasp.')
            return None, None

        cloud_frame = self.point_cloud.header.frame_id
        target_cloud = self._transform_point(
            np.array([target_msg.point.x, target_msg.point.y, target_msg.point.z], dtype=float),
            target_msg.header.frame_id,
            cloud_frame,
        )
        target_base = self._transform_point(
            np.array([target_msg.point.x, target_msg.point.y, target_msg.point.z], dtype=float),
            target_msg.header.frame_id,
            'base_link',
        )
        if target_cloud is None or target_base is None:
            self.logger.error('Unable to transform target point for top-down grasp planning.')
            return None, None

        radius_sq = float(self.object_cluster_radius_m) ** 2
        vertical_window_m = float(self.object_vertical_window_m)
        local_points_cloud = []

        for point in point_cloud2.read_points(
            self.point_cloud,
            field_names=('x', 'y', 'z'),
            skip_nans=True,
        ):
            dx = float(point[0]) - float(target_cloud[0])
            dy = float(point[1]) - float(target_cloud[1])
            dz = float(point[2]) - float(target_cloud[2])
            if (dx * dx) + (dy * dy) > radius_sq:
                continue
            if abs(dz) > vertical_window_m:
                continue
            local_points_cloud.append((float(point[0]), float(point[1]), float(point[2])))

        if len(local_points_cloud) < self.min_object_points:
            self.logger.warn(
                f'Only found {len(local_points_cloud)} local point-cloud samples near the target.'
            )
            return target_base, None

        local_points_base = self._transform_points(local_points_cloud, cloud_frame, 'base_link')
        if local_points_base is None:
            self.logger.error('Unable to transform local point-cloud crop into base_link.')
            return target_base, None

        return target_base, local_points_base

    def _estimate_object_footprint(self, target_base, local_points_base, fallback_width_m):
        if local_points_base is None or len(local_points_base) < self.min_object_points:
            return None

        points = np.asarray(local_points_base, dtype=float)
        floor_z = float(np.percentile(points[:, 2], 10.0))
        object_mask = (
            (points[:, 2] > (floor_z + self.min_object_height_above_floor_m))
            & (points[:, 2] < (floor_z + self.max_object_height_m))
        )
        object_points = points[object_mask]
        if len(object_points) < max(20, self.min_object_points // 3):
            self.logger.warn('Unable to isolate enough above-floor points for top-down grasping.')
            return None

        target_xy = np.asarray(target_base[:2], dtype=float)
        xy = object_points[:, :2]
        center_xy = xy.mean(axis=0)
        centered_xy = xy - center_xy

        if len(centered_xy) < 3:
            return None

        covariance = np.cov(centered_xy, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        major_axis = eigenvectors[:, 0]
        major_axis /= max(np.linalg.norm(major_axis), 1e-6)
        minor_axis = np.array([-major_axis[1], major_axis[0]], dtype=float)

        projections_major = centered_xy @ major_axis
        projections_minor = centered_xy @ minor_axis
        major_length_m = float(np.percentile(projections_major, 95.0) - np.percentile(projections_major, 5.0))
        minor_length_m = float(np.percentile(projections_minor, 95.0) - np.percentile(projections_minor, 5.0))
        elongation_ratio = major_length_m / max(minor_length_m, 1e-3)
        elongated = bool(elongation_ratio >= self.min_elongation_ratio)

        center_target_delta = float(np.linalg.norm(center_xy - target_xy))
        if center_target_delta > self.object_cluster_radius_m:
            self.logger.warn(
                f'Estimated object footprint center is {center_target_delta:.3f} m from the requested target.'
            )

        top_z = float(np.percentile(object_points[:, 2], 95.0))
        centroid_z = float(np.mean(object_points[:, 2]))
        target_z = float(target_base[2])
        grasp_tip_z = max(target_z, self.topdown_floor_clearance_m)
        hover_tip_z = grasp_tip_z + max(self.topdown_hover_above_object_m, 0.03)

        return {
            'center_xy': center_xy,
            'floor_z': floor_z,
            'top_z': top_z,
            'centroid_z': centroid_z,
            'grasp_tip_z': grasp_tip_z,
            'hover_tip_z': hover_tip_z,
            'major_axis_xy': major_axis,
            'major_length_m': max(major_length_m, fallback_width_m),
            'minor_length_m': max(minor_length_m, fallback_width_m),
            'elongated': elongated,
            'elongation_ratio': elongation_ratio,
        }

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

    def _compute_hover_pose(self, target_xy, hover_tip_z):
        tip_base = self._get_gripper_tip_in_frame('base_link')
        if tip_base is None or self.wrist_position is None or self.lift_position is None:
            return None

        delta_base_m = float(target_xy[0] - tip_base[0])
        delta_extension_m = -float(target_xy[1] - tip_base[1])
        desired_extension_m = float(self.wrist_position) + delta_extension_m
        desired_lift_m = float(self.lift_position) + float(hover_tip_z - tip_base[2])

        return delta_base_m, desired_extension_m, desired_lift_m

    def _compute_lift_for_tip_height(self, desired_tip_z):
        tip_base = self._get_gripper_tip_in_frame('base_link')
        if tip_base is None or self.lift_position is None:
            return None
        return float(self.lift_position) + float(desired_tip_z - tip_base[2])

    def grasp_point_service_callback(self, request, response):
        try:
            if self.get_tool() != 'tool_stretch_dex_wrist':
                response.success = False
                response.message = 'Top-down grasping requires tool_stretch_dex_wrist.'
                self.logger.error(response.message)
                return response

            msg = request.target
            assumed_width_m = (
                float(request.assumed_width_m)
                if request.assumed_width_m > 0.0
                else float(self.assumed_width_m)
            )

            if msg.header.stamp.sec == 0 and msg.header.stamp.nanosec == 0:
                self.logger.error('Target point timestamp is unset, so unable to grasp.')
                response.success = False
                response.message = 'Target point timestamp is unset, so unable to grasp.'
                return response

            if self.lift_position is None or self.wrist_position is None:
                self.logger.error('Joint state information is unavailable.')
                response.success = False
                response.message = 'Joint state information is unavailable.'
                return response

            self.logger.info(
                'Top-down grasp point received: '
                f'frame={msg.header.frame_id} '
                f'x={msg.point.x:.3f} y={msg.point.y:.3f} z={msg.point.z:.3f}'
            )

            self.stow_the_robot()

            ## TODO delete unnecesary for topdown.
            manip = self.look_at_surface(scan_time_s=self.scan_time_s)
            if manip is None:
                self.logger.error('Failed to build manipulation view.')
                response.success = False
                response.message = 'Failed to build manipulation view.'
                return response

            target_base, local_points_base = self._collect_local_points_in_base_frame(msg)
            if target_base is None:
                self.logger.error('Unable to transform the target point into base_link.')
                response.success = False
                response.message = 'Unable to transform the target point into base_link.'
                return response

            footprint = self._estimate_object_footprint(target_base, local_points_base, assumed_width_m)
            if footprint is None:
                self.logger.error('Unable to estimate a local object footprint for top-down grasping.')
                response.success = False
                response.message = 'Unable to estimate a local object footprint for top-down grasping.'
                return response

            topdown_roll = self._compute_topdown_roll(footprint['major_axis_xy'])
            self.logger.info(
                'Estimated top-down grasp footprint: '
                f'elongated={footprint["elongated"]} '
                f'ratio={footprint["elongation_ratio"]:.2f} '
                f'major={footprint["major_length_m"]:.3f} m '
                f'minor={footprint["minor_length_m"]:.3f} m '
                f'top_z={footprint["top_z"]:.3f} m '
                f'grasp_roll={topdown_roll:.3f} rad'
            )

            max_lift_m = 1.09
            min_lift_m = 0.07
            min_extension_m = 0.01
            max_extension_m = 0.5

            prealign_lift_m = max(float(self.prealign_lift_m), min_lift_m)
            prealign_lift_m = min(prealign_lift_m, max_lift_m)
            self.logger.info('Raise the lift to a safe prealignment height.')
            self.debug_move_to_pose({'joint_lift': prealign_lift_m})

            initial_extension_m = max(float(self.initial_wrist_extension_m), min_extension_m)
            initial_extension_m = min(initial_extension_m, max_extension_m)
            self.logger.info('Extend the wrist slightly before pitching downward.')
            self.debug_move_to_pose({'wrist_extension': initial_extension_m})

            self.logger.info('Pitch the wrist into the top-down grasp orientation.')
            self.debug_move_to_pose(
                {
                    'joint_wrist_pitch': float(self.topdown_wrist_pitch_rad),
                    'joint_wrist_yaw': float(self.topdown_wrist_yaw_rad),
                }
            )

            self.logger.info('Rotate the wrist roll to align with the object long axis.')
            self.debug_move_to_pose({'joint_wrist_roll': topdown_roll})

            hover_pose = self._compute_hover_pose(target_base[:2], footprint['hover_tip_z'])
            if hover_pose is None:
                self.logger.error('Unable to compute the hover pose for top-down grasping.')
                response.success = False
                response.message = 'Unable to compute the hover pose for top-down grasping.'
                return response

            hover_base_delta_m, hover_extension_m, hover_lift_m = hover_pose
            hover_extension_m = max(hover_extension_m, min_extension_m)
            hover_extension_m = min(hover_extension_m, max_extension_m)
            hover_lift_m = max(hover_lift_m, min_lift_m)
            hover_lift_m = min(hover_lift_m, max_lift_m)

            self.logger.info('Move the gripper above the requested grasp point.')
            self.debug_move_to_pose(
                {
                    'translate_mobile_base': hover_base_delta_m,
                    'wrist_extension': hover_extension_m,
                    'joint_lift': hover_lift_m,
                }
            )

            self.logger.info('Open the gripper before the vertical descent.')
            self.debug_move_to_pose({'gripper_aperture': 0.77})

            grasp_lift_m = self._compute_lift_for_tip_height(footprint['grasp_tip_z'])
            if grasp_lift_m is None:
                self.logger.error('Unable to compute the final top-down grasp height.')
                response.success = False
                response.message = 'Unable to compute the final top-down grasp height.'
                return response

            grasp_lift_m = max(grasp_lift_m, min_lift_m)
            grasp_lift_m = min(grasp_lift_m, max_lift_m)

            self.logger.info('Descend vertically to the grasp height.')
            self.debug_move_to_pose({'joint_lift': grasp_lift_m})

            self.logger.info('Close the gripper on the object.')
            self.debug_move_to_pose({'gripper_aperture': 0.0})
            time.sleep(2.0)

            lift_with_object_m = max(float(self.lift_position) + self.object_lift_height_m, 0.2)
            lift_with_object_m = min(lift_with_object_m, max_lift_m)
            self.logger.info('Lift the object after grasping.')
            self.debug_move_to_pose({'joint_lift': lift_with_object_m})

            self.logger.info('Open the gripper slightly to avoid overtorquing.')
            self.debug_move_to_pose({'gripper_aperture': 0.005})

            self.logger.info('Restore the wrist orientation before retracting.')
            self.debug_move_to_pose(
                {
                    'joint_wrist_pitch': 0.0,
                    'joint_wrist_roll': 0.0,
                    'joint_wrist_yaw': 0.0,
                }
            )
            self.logger.info('Retract the wrist.')
            self.debug_move_to_pose({'wrist_extension': 0.01})

            self._clear_translation_marker()
            response.success = True
            response.message = 'Completed top-down object grasp!'
            return response
        except DebugMoveCancelled as exc:
            self._clear_translation_marker()
            self.logger.warn(str(exc))
            response.success = False
            response.message = str(exc)
            return response

    def main(self):
        hm.HelloNode.main(
            self,
            'topdown_grasp_point',
            'topdown_grasp_point',
            wait_for_first_pointcloud=False,
        )
        self.logger = self.get_logger()

        self.callback_group = ReentrantCallbackGroup()

        self.declare_parameter('debug_directory', '')
        self.declare_parameter('assumed_width_m', 0.05)
        self.declare_parameter('scan_time_s', 2.0)
        self.declare_parameter('grasp_service', '/topdown_grasp_point')
        self.declare_parameter('dryrun', False)
        self.declare_parameter('debug_mode', False)
        self.declare_parameter('object_cluster_radius_m', 0.12)
        self.declare_parameter('object_vertical_window_m', 0.20)
        self.declare_parameter('min_object_height_above_floor_m', 0.01)
        self.declare_parameter('max_object_height_m', 0.25)
        self.declare_parameter('min_object_points', 60)
        self.declare_parameter('prealign_lift_m', 0.85)
        self.declare_parameter('initial_wrist_extension_m', 0.12)
        self.declare_parameter('topdown_wrist_pitch_rad', -math.pi / 2.0 + 0.07)
        self.declare_parameter('topdown_wrist_yaw_rad', 0.0)
        self.declare_parameter('topdown_hover_above_object_m', 0.35)
        self.declare_parameter('topdown_floor_clearance_m', 0.025)
        self.declare_parameter('topdown_below_top_m', 0.01)
        self.declare_parameter('object_lift_height_m', 0.10)
        self.declare_parameter('roll_axis_offset_rad', 0.0)
        self.declare_parameter('min_elongation_ratio', 1.35)

        self.debug_directory = self.get_parameter('debug_directory').value
        self.assumed_width_m = float(self.get_parameter('assumed_width_m').value)
        self.scan_time_s = float(self.get_parameter('scan_time_s').value)
        grasp_service = self.get_parameter('grasp_service').value

        self.dryrun = self.get_parameter('dryrun').value
        self.debug_mode = bool(self.get_parameter('debug_mode').value)
        self.object_cluster_radius_m = float(self.get_parameter('object_cluster_radius_m').value)
        self.object_vertical_window_m = float(self.get_parameter('object_vertical_window_m').value)
        self.min_object_height_above_floor_m = float(
            self.get_parameter('min_object_height_above_floor_m').value
        )
        self.max_object_height_m = float(self.get_parameter('max_object_height_m').value)
        self.min_object_points = int(self.get_parameter('min_object_points').value)
        self.prealign_lift_m = float(self.get_parameter('prealign_lift_m').value)
        self.initial_wrist_extension_m = float(self.get_parameter('initial_wrist_extension_m').value)
        self.topdown_wrist_pitch_rad = float(self.get_parameter('topdown_wrist_pitch_rad').value)
        self.topdown_wrist_yaw_rad = float(self.get_parameter('topdown_wrist_yaw_rad').value)
        self.topdown_hover_above_object_m = float(
            self.get_parameter('topdown_hover_above_object_m').value
        )
        self.topdown_floor_clearance_m = float(
            self.get_parameter('topdown_floor_clearance_m').value
        )
        self.topdown_below_top_m = float(self.get_parameter('topdown_below_top_m').value)
        self.object_lift_height_m = float(self.get_parameter('object_lift_height_m').value)
        self.roll_axis_offset_rad = float(self.get_parameter('roll_axis_offset_rad').value)
        self.min_elongation_ratio = float(self.get_parameter('min_elongation_ratio').value)

        self.joint_states_subscriber = self.create_subscription(
            JointState,
            '/stretch/joint_states',
            callback=self.joint_states_callback,
            qos_profile=1,
            callback_group=self.callback_group,
        )

        self.grasp_point_service = self.create_service(
            GraspPoint,
            grasp_service,
            self.grasp_point_service_callback,
            callback_group=self.callback_group,
        )

        self.debug_marker_pub = self.create_publisher(
            Marker,
            '/topdown_grasp_point/debug_translation_marker',
            1,
        )

        self.logger.info(f'Top-down grasp point node is ready. Service: {grasp_service}')


def main():
    try:
        node = TopdownGraspPointNode()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        rclpy.logging.get_logger('topdown_grasp_point').info(
            'interrupt received, so shutting down'
        )


if __name__ == '__main__':
    main()
