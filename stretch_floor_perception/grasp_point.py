#!/usr/bin/env python3

from geometry_msgs.msg import Point, PointStamped
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
import rclpy.logging

import time
import threading
import numpy as np
import os

import hello_helpers.hello_misc as hm
import stretch_funmap.navigate as nv
import stretch_funmap.manipulation_planning as mp
from stretch_floor_perception.srv import GraspPoint


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
            dirname = self.debug_directory + 'grasp_point/'
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
        marker.ns = 'grasp_point_debug'
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.lifetime = Duration(seconds=30.0).to_msg()
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
        marker.ns = 'grasp_point_debug'
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

    def debug_stow_the_robot(self, description='stow the robot'):
        self._confirm_motion(description)
        self.stow_the_robot()

    def grasp_point_service_callback(self, request, response):
        try:
            msg = request.target
            assumed_width_m = request.assumed_width_m if request.assumed_width_m > 0.0 else self.assumed_width_m

            if self.header.stamp.sec > 0:
                self.logger.error('No point cloud available yet, so unable to grasp.')
                response.success = False
                response.message = 'No point cloud available yet, so unable to grasp.'
                return response

            self.logger.info(f'Grasp point received: frame={msg.header.frame_id} x={msg.point.x:.3f} y={msg.point.y:.3f} z={msg.point.z:.3f}')

            # Optional: stow before the attempt to get a consistent starting pose.
            self.debug_stow_the_robot()

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
            min_extension_m = 0.01
            max_extension_m = 0.5
            initial_wrist_extension = 0.2

            pregrasp_lift_m = manip.get_pregrasp_lift(grasp_target, self.tf2_buffer)
            if self.get_tool() == "tool_stretch_dex_wrist":
                pregrasp_lift_m += 0.02
            if self.lift_position is None:
                self.logger.error('lift position unavailable')
                response.success = False
                response.message = 'lift position unavailable'
                return response

            if self.get_tool() == "tool_stretch_dex_wrist":
                self.logger.info('Rotate pitch/roll for grasping.')
                self.debug_move_to_pose({'joint_wrist_pitch': -0.3, 'joint_wrist_roll': 0.0})

            pregrasp_yaw = manip.get_pregrasp_yaw(grasp_target, self.tf2_buffer)
            self.logger.info('Rotate the gripper for grasping.')
            self.debug_move_to_pose({'joint_wrist_yaw': pregrasp_yaw})

            self.logger.info('Extend wrist to avoid collision with base.')
            self.debug_move_to_pose({'wrist_extension': initial_wrist_extension})

            self.logger.info('Raise tool to pregrasp height.')
            lift_to_pregrasp_m = max(self.lift_position + pregrasp_lift_m, 0.1) # TODO fix, it's always max since it's negative
            lift_to_pregrasp_m = min(lift_to_pregrasp_m, max_lift_m)
            self.debug_move_to_pose({'joint_lift': lift_to_pregrasp_m})

            self.logger.info('Open the gripper.')
            self.debug_move_to_pose({'gripper_aperture': 0.07})

            pregrasp_mobile_base_m, pregrasp_wrist_extension_m = (
                manip.get_pregrasp_planar_translation(grasp_target, self.tf2_buffer)
            )

            self.logger.info(f"debug pregrasp_mobile_base: {pregrasp_mobile_base_m}\n debug pregrasp_wrist_extension: {pregrasp_wrist_extension_m}")

            self.logger.info('Drive to pregrasp location.')
            self.debug_drive(pregrasp_mobile_base_m)

            if self.wrist_position is None:
                self.logger.error('wrist position unavailable')
                response.success = False
                response.message = 'wrist position unavailable'
                return response

            if pregrasp_wrist_extension_m > 0.0: 
                extension_m = max(self.wrist_position + pregrasp_wrist_extension_m, min_extension_m)
                extension_m = min(extension_m, max_extension_m)
                self.logger.info('Extend tool above surface.')
                self.debug_move_to_pose({'wrist_extension': extension_m})

            grasp_mobile_base_m, grasp_lift_m, grasp_wrist_extension_m = (
                manip.get_grasp_from_pregrasp(grasp_target, self.tf2_buffer)
            )
            self.logger.info('Move the grasp pose from the pregrasp pose.')

            lift_m = max(self.lift_position + grasp_lift_m, 0.1)
            lift_m = min(lift_m, max_lift_m)
            extension_m = max(self.wrist_position + grasp_wrist_extension_m, min_extension_m)
            extension_m = min(extension_m, max_extension_m)
            self.debug_move_to_pose({
                'translate_mobile_base': grasp_mobile_base_m,
                'joint_lift': lift_m,
                'wrist_extension': extension_m
            })

            self.logger.info('Attempt to close the gripper on the object.')
            gripper_aperture_m = max(0.0, assumed_width_m - 0.18)
            self.debug_move_to_pose({'gripper_aperture': gripper_aperture_m})

            time.sleep(3.0)

            self.logger.info('Attempt to lift the object.')
            object_lift_height_m = 0.1
            lift_m = max(self.lift_position + object_lift_height_m, 0.2)
            lift_m = min(lift_m, max_lift_m)
            self.debug_move_to_pose({'joint_lift': lift_m})

            self.logger.info('Open the gripper a little to avoid overtorquing and overheating the gripper motor.')
            self.debug_move_to_pose({'gripper_aperture': gripper_aperture_m + 0.005})

            self.logger.info('Retract the tool.')
            self.debug_move_to_pose({'wrist_extension': 0.01})

            self.logger.info('Reorient the wrist.')
            self.debug_move_to_pose({'joint_wrist_yaw': 0.0})

            self._clear_translation_marker()
            response.success = True
            response.message = 'Completed object grasp!'
            return response
        except DebugMoveCancelled as exc:
            self._clear_translation_marker()
            self.logger.warn(str(exc))
            response.success = False
            response.message = str(exc)
            return response

    def main(self):
        hm.HelloNode.main(self, 'grasp_point', 'grasp_point', wait_for_first_pointcloud=False)
        self.logger = self.get_logger()

        self.callback_group = ReentrantCallbackGroup()

        self.declare_parameter('debug_directory', '')
        self.declare_parameter('assumed_width_m', 0.05)
        self.declare_parameter('scan_time_s', 2.0)
        self.declare_parameter('grasp_service', '/grasp_point')
        self.declare_parameter('dryrun', False)
        self.declare_parameter('debug_mode', False)

        self.debug_directory = self.get_parameter('debug_directory').value
        self.assumed_width_m = float(self.get_parameter('assumed_width_m').value)
        self.scan_time_s = float(self.get_parameter('scan_time_s').value)
        grasp_service = self.get_parameter('grasp_service').value

        self.dryrun = self.get_parameter('dryrun').value
        self.debug_mode = bool(self.get_parameter('debug_mode').value)

        self.joint_states_subscriber = self.create_subscription(
            JointState, '/stretch/joint_states', callback=self.joint_states_callback, qos_profile=1, callback_group=self.callback_group)

        self.grasp_point_service = self.create_service(
            GraspPoint, grasp_service, self.grasp_point_service_callback, callback_group=self.callback_group)

        self.debug_marker_pub = self.create_publisher(
            Marker, '/grasp_point/debug_translation_marker', 1)

        self.logger.info(f'Grasp point node is ready. Service: {grasp_service}')


def main():
    try:
        node = GraspPointNode()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        rclpy.logging.get_logger('grasp_point').info('interrupt received, so shutting down')


if __name__ == '__main__':
    main()
