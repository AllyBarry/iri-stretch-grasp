#!/usr/bin/env python3

# Handover procedure for Stretch (dex wrist):
#   1. Present the held object: raise the lift, extend the arm slightly and
#      point the head camera at the gripper.
#   2. Learn what the gripper + object look like in the head camera point cloud.
#   3. Wait for a hand to appear on the object (new 3D points around the
#      gripper that persist for release_delay_s).
#   4. Open the gripper, wait for the hand to clear, then stow.

from std_srvs.srv import Trigger
from sensor_msgs_py import point_cloud2

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
import rclpy.logging

import time
import threading
import numpy as np

import hello_helpers.hello_misc as hm


class DebugMoveCancelled(RuntimeError):
    pass


class HandoverNode(hm.HelloNode):

    def __init__(self):
        hm.HelloNode.__init__(self)
        self.handover_lock = threading.Lock()
        self.debug_prompt_lock = threading.Lock()
        self.last_cloud_stamp = None

    # ------------------------------------------------------------------ motion

    def _confirm_motion(self, description):
        if (not self.debug_mode) or self.dryrun:
            return
        try:
            with self.debug_prompt_lock:
                answer = input(f'{description}. Proceed? [y/N]: ').strip().lower()
        except EOFError:
            answer = ''
        if answer not in ('y', 'yes'):
            raise DebugMoveCancelled(f'User declined motion: {description}')

    def debug_move_to_pose(self, pose, description):
        self.logger.info(description)
        self._confirm_motion(description)
        return self.move_to_pose(pose)

    def _point_head_at_gripper(self):
        if self.dryrun:
            return
        # Give TF a moment to reflect the new arm pose.
        time.sleep(0.3)
        tip_to_base_mat, _ = hm.get_p1_to_p2_matrix('link_grasp_center', 'base_link', self.tf2_buffer, timeout_s=1.0)
        head_to_base_mat, _ = hm.get_p1_to_p2_matrix('link_head_tilt', 'base_link', self.tf2_buffer, timeout_s=1.0)
        if (tip_to_base_mat is None) or (head_to_base_mat is None):
            self.logger.warn('Unable to locate the gripper relative to the head, so not aiming the camera.')
            return
        dx, dy, dz = tip_to_base_mat[:3, 3] - head_to_base_mat[:3, 3]
        pan = float(np.arctan2(dy, dx))
        tilt = float(np.arctan2(dz, np.hypot(dx, dy)))
        self.debug_move_to_pose({'joint_head_pan': pan, 'joint_head_tilt': tilt},
                                f'Point the head camera at the gripper (pan={pan:.3f}, tilt={tilt:.3f})')

    def present_object(self):
        self.debug_move_to_pose({'joint_lift': self.present_lift_m},
                                f'Raise the lift to {self.present_lift_m:.2f} m')

        wrist_pose = {'joint_wrist_yaw': self.present_wrist_yaw_rad}
        if self.get_tool() == 'tool_stretch_dex_wrist':
            wrist_pose['joint_wrist_pitch'] = self.present_wrist_pitch_rad
            wrist_pose['joint_wrist_roll'] = 0.0
        self.debug_move_to_pose(wrist_pose, 'Orient the wrist for the handover')

        self.debug_move_to_pose({'wrist_extension': self.present_extension_m},
                                f'Extend the arm to {self.present_extension_m:.2f} m')

        self._point_head_at_gripper()

    # --------------------------------------------------------------- perception

    def _wait_for_new_cloud(self, timeout_s=1.0):
        start_s = time.time()
        while (time.time() - start_s) < timeout_s:
            msg = self.point_cloud
            stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            if (msg.header.stamp.sec > 0) and (stamp != self.last_cloud_stamp):
                self.last_cloud_stamp = stamp
                return msg
            time.sleep(0.02)
        return None

    def _occupied_voxels(self):
        # Returns the set of voxel keys occupied by head camera points within
        # contact_radius_m of the grasp center, or None if no cloud/TF.
        msg = self._wait_for_new_cloud()
        if msg is None:
            self.logger.warn('No new point cloud received from /camera/depth/color/points.')
            return None

        cloud_to_grasp_mat, _ = hm.get_p1_to_p2_matrix(msg.header.frame_id, 'link_grasp_center', self.tf2_buffer)
        if cloud_to_grasp_mat is None:
            return None

        xyz = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        xyz = xyz[::self.cloud_stride].astype(np.float64)
        xyz = xyz @ cloud_to_grasp_mat[:3, :3].T + cloud_to_grasp_mat[:3, 3]
        xyz = xyz[np.linalg.norm(xyz, axis=1) < self.contact_radius_m]

        ijk = np.floor(xyz / self.voxel_size_m).astype(np.int64) + 512
        keys = (ijk[:, 0] << 20) | (ijk[:, 1] << 10) | ijk[:, 2]
        return set(np.unique(keys).tolist())

    def _learn_baseline(self):
        # The gripper and held object are static during the handover, so
        # anything occupying voxels outside this (dilated) baseline is new.
        self.logger.info(f'Learning the gripper/object appearance for {self.baseline_s:.1f} s. Keep hands away.')
        baseline = set()
        num_clouds = 0
        start_s = time.time()
        while (time.time() - start_s) < self.baseline_s:
            voxels = self._occupied_voxels()
            if voxels is not None:
                baseline |= voxels
                num_clouds += 1
        if num_clouds == 0:
            return None

        offsets = [(dx << 20) + (dy << 10) + dz
                   for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        dilated = {key + offset for key in baseline for offset in offsets}
        self.logger.info(f'Baseline: {len(baseline)} voxels from {num_clouds} clouds.')
        return dilated

    def _hand_present(self, baseline):
        voxels = self._occupied_voxels()
        if voxels is None:
            return None
        num_new = len(voxels - baseline)
        self.logger.debug(f'{num_new} new voxels near the gripper.')
        return num_new >= self.min_new_voxels

    def wait_for_hand(self, baseline):
        # A hand counts once it has been continuously present for
        # release_delay_s; if it leaves early, the wait starts over.
        self.logger.info('Waiting for a hand on the object.')
        start_s = time.time()
        present_count = 0
        hand_since_s = None
        while (time.time() - start_s) < self.hand_timeout_s:
            present = self._hand_present(baseline)
            if present is None:
                continue
            present_count = present_count + 1 if present else 0

            if present_count >= self.min_consecutive_frames:
                if hand_since_s is None:
                    hand_since_s = time.time()
                    self.logger.info(f'Hand detected. Releasing in {self.release_delay_s:.1f} s if it stays.')
                elif (time.time() - hand_since_s) >= self.release_delay_s:
                    return True
            elif hand_since_s is not None:
                self.logger.info('Hand left before the release delay elapsed, so waiting again.')
                hand_since_s = None
        return False

    def wait_for_hand_clear(self, baseline):
        # Avoid stowing while the person's fingers are still in the gripper.
        start_s = time.time()
        clear_count = 0
        while (time.time() - start_s) < self.clear_timeout_s:
            present = self._hand_present(baseline)
            clear_count = clear_count + 1 if present is False else 0
            if clear_count >= self.min_consecutive_frames:
                return True
        return False

    # ---------------------------------------------------------------- procedure

    def handover(self):
        if not self.handover_lock.acquire(blocking=False):
            return False, 'A handover is already in progress.'
        try:
            self.present_object()

            if self.dryrun:
                return True, 'Dry run: skipped hand detection and release.'

            baseline = self._learn_baseline()
            if baseline is None:
                return False, 'Unable to learn a baseline: no point cloud or TF available.'

            if not self.wait_for_hand(baseline):
                self.debug_move_to_pose({'wrist_extension': 0.01},
                                        'No hand detected, so retract the arm while keeping the object')
                return False, f'No hand detected within {self.hand_timeout_s:.0f} s.'

            self.debug_move_to_pose({'gripper_aperture': self.release_aperture_m},
                                    'Open the gripper to release the object')

            if not self.wait_for_hand_clear(baseline):
                self.logger.warn(f'Hand still near the gripper after {self.clear_timeout_s:.0f} s; stowing anyway.')

            self.logger.info('Stow the robot.')
            self._confirm_motion('Stow the robot')
            self.stow_the_robot()
            return True, 'Object handed over.'
        except DebugMoveCancelled as exc:
            self.logger.warn(str(exc))
            return False, str(exc)
        finally:
            self.handover_lock.release()

    def handover_service_callback(self, request, response):
        response.success, response.message = self.handover()
        return response

    def _declare_parameter_once(self, name, default):
        # HelloNode auto-declares parameters passed as overrides (--params-file, -p),
        # so only declare the ones that were not overridden.
        if not self.has_parameter(name):
            self.declare_parameter(name, default)

    def main(self):
        hm.HelloNode.main(self, 'handover', 'handover', wait_for_first_pointcloud=False)
        self.logger = self.get_logger()

        defaults = {
            'present_lift_m': 0.85,
            'present_extension_m': 0.20,
            'present_wrist_yaw_rad': 0.0,
            'present_wrist_pitch_rad': 0.0,
            'release_aperture_m': 0.07,
            'contact_radius_m': 0.15,
            'voxel_size_m': 0.015,
            'cloud_stride': 2,
            'baseline_s': 1.5,
            'min_new_voxels': 20,
            'min_consecutive_frames': 3,
            'release_delay_s': 2.0,
            'hand_timeout_s': 60.0,
            'clear_timeout_s': 5.0,
        }
        for name, default in defaults.items():
            self._declare_parameter_once(name, default)
            value = self.get_parameter(name).value
            setattr(self, name, int(value) if isinstance(default, int) else float(value))

        self._declare_parameter_once('dryrun', False)
        self._declare_parameter_once('debug_mode', False)
        self._declare_parameter_once('run_once', True)
        self._declare_parameter_once('handover_service', '/handover')
        self.dryrun = bool(self.get_parameter('dryrun').value)
        self.debug_mode = bool(self.get_parameter('debug_mode').value)

        if self.get_parameter('run_once').value:
            # Wait for /tool so get_tool() is meaningful.
            while not isinstance(self.tool, str):
                time.sleep(0.1)
            success, message = self.handover()
            (self.logger.info if success else self.logger.error)(message)
            return

        handover_service = self.get_parameter('handover_service').value
        self.handover_service = self.create_service(
            Trigger, handover_service, self.handover_service_callback, callback_group=ReentrantCallbackGroup())
        self.logger.info(f'Handover node is ready. Service: {handover_service}')
        self.new_thread.join()


def main():
    try:
        node = HandoverNode()
        node.main()
    except KeyboardInterrupt:
        rclpy.logging.get_logger('handover').info('interrupt received, so shutting down')


if __name__ == '__main__':
    main()
