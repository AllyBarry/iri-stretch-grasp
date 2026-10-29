#!/usr/bin/env python3
"""place_object.py: a /place_object service that sets a held object down on the surface below it.

  ros2 run stretch_floor_perception place_object.py
  ros2 service call /place_object std_srvs/srv/Trigger {}

The gripper is assumed to be over the place to put the object already: the service only
lowers, releases and lifts clear. It looks at the gripper with the head camera and measures
the surface under it and the bottom of the held object, then lowers quickly until the object
is approach_gap_m above the surface. From there it steps down step_m at a time and stops when
the lift effort drops by contact_effort_drop_pct (the surface is taking the object's weight),
when the object bottom is press_m below the measured surface, or before the fingertips come
within finger_clearance_m of the surface, whichever comes first. Then it opens the gripper and
raises the lift by raise_after_m. The base, arm extension and wrist are not moved.
"""

import time

import numpy as np
import rclpy
import rclpy.logging
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker

import hello_helpers.hello_misc as hm
from stretch_floor_perception.grasp_carry import DebugMoveCancelled, GraspPointNode


class PlaceObjectNode(GraspPointNode):

    def __init__(self):
        super().__init__()
        # The surface may be well below the gripper, so take points from far below it.
        self.object_vertical_window_m = 1.2
        # Ring around the gripper to find the surface in: close in first, wider if too few points.
        self.surface_ring_m = (0.08, 0.15, 0.25)
        self.object_column_radius_m = 0.08
        self.approach_gap_m = 0.03
        # Gap to stop the fast move at when the held object cannot be seen (its length unknown).
        self.unseen_object_gap_m = 0.20
        self.step_m = 0.01
        self.contact_effort_drop_pct = 8.0
        self.press_m = 0.02
        self.finger_clearance_m = 0.01
        self.release_aperture_m = 0.07
        self.release_wait_s = 1.0
        self.raise_after_m = 0.05
        self.effort_samples = 5

    # ------------------------------------------------------------------
    # Measuring the surface and the held object
    # ------------------------------------------------------------------

    def _gripper_bottom_z(self):
        """Lowest of the grasp centre and the two fingertips, in base_link."""
        zs = []
        for frame in ('link_grasp_center', 'link_gripper_fingertip_left', 'link_gripper_fingertip_right'):
            p = self._get_frame_in_base(frame)
            if p is not None:
                zs.append(float(p[2]))
        return min(zs) if zs else None

    def _measure_surface_and_object(self, grasp_base, gripper_bottom_z):
        """Returns (surface_z, object_bottom_z or None) in base_link, or (None, None).

        The surface is the highest well-supported height in a ring close around the gripper
        (the table rather than the floor beyond its edge). The object bottom is the lowest of
        the points hanging in a column under the gripper, above that surface.
        """
        point_cloud_msg = self._wait_for_fresh_point_cloud()
        if point_cloud_msg is None:
            self.logger.error('No fresh point cloud to measure the surface.')
            return None, None
        points = self._collect_local_points_in_base_frame(
            point_cloud_msg, grasp_base, self.surface_ring_m[-1])
        if points is None or len(points) == 0:
            self.logger.error('No points around the gripper to measure the surface.')
            return None, None
        r = np.hypot(points[:, 0] - grasp_base[0], points[:, 1] - grasp_base[1])

        below = points[:, 2] < gripper_bottom_z - 0.02
        for outer in self.surface_ring_m[1:]:
            ring = points[(r >= self.surface_ring_m[0]) & (r < outer) & below]
            if len(ring) >= 50:
                break
        else:
            self.logger.error(f'Only {len(ring)} points on the surface around the gripper.')
            return None, None
        bins = np.floor(ring[:, 2] / 0.01).astype(int)
        values, counts = np.unique(bins, return_counts=True)
        # Horizontal surfaces pile up in one height bin; the highest well-filled one is the
        # surface under the gripper, even with more floor in view past the table edge.
        supported = values[counts >= max(30, 0.15 * counts.max())]
        surface_bin = supported.max()
        on_surface = ring[np.abs(bins - surface_bin) <= 1]
        surface_z = float(np.median(on_surface[:, 2]))

        column = points[(r < self.object_column_radius_m)
                        & (points[:, 2] > surface_z + 0.01) & (points[:, 2] < gripper_bottom_z + 0.05)]
        object_bottom_z = float(np.percentile(column[:, 2], 2.0)) if len(column) >= 15 else None
        return surface_z, object_bottom_z

    def _lift_effort(self):
        """Mean lift effort (motor effort percent) over a few joint state messages."""
        samples = []
        for _ in range(self.effort_samples):
            with self.joint_states_lock:
                joint_states = self.joint_states
            if joint_states is not None and 'joint_lift' in joint_states.name and joint_states.effort:
                samples.append(float(joint_states.effort[joint_states.name.index('joint_lift')]))
            time.sleep(0.05)
        return float(np.mean(samples)) if samples else None

    # ------------------------------------------------------------------
    # Service
    # ------------------------------------------------------------------

    def _fail(self, response, message):
        self.logger.error(message)
        response.success = False
        response.message = message
        return response

    def place_object_service_callback(self, request, response):
        try:
            if self.lift_position is None:
                return self._fail(response, 'Joint state information is unavailable.')
            max_lift_m = 1.09
            min_lift_m = 0.03

            grasp_base = self._get_frame_in_base('link_grasp_center')
            gripper_bottom_z = self._gripper_bottom_z()
            if grasp_base is None or gripper_bottom_z is None:
                return self._fail(response, 'Unable to locate the gripper.')

            # 1. Look at the gripper and measure the surface under it and the held object.
            self._point_head_at(grasp_base - np.array([0.0, 0.0, 0.1]), 'the gripper and the surface below')
            time.sleep(self.head_settle_s)
            surface_z, object_bottom_z = self._measure_surface_and_object(grasp_base, gripper_bottom_z)
            if surface_z is None:
                return self._fail(response, 'Unable to measure the surface under the gripper, so not placing.')
            object_seen = object_bottom_z is not None
            if not object_seen:
                self.logger.warn(
                    'Could not see the held object, so lowering fast only to '
                    f'{100 * self.unseen_object_gap_m:.0f} cm above the surface and stepping down from there.')
                object_bottom_z = gripper_bottom_z
            self.logger.info(
                f'Surface at {surface_z:.3f} m, held object bottom at {object_bottom_z:.3f} m '
                f'({100 * (object_bottom_z - surface_z):.1f} cm above it), fingertips at {gripper_bottom_z:.3f} m.')

            # The lowest the lift may go: fingertips finger_clearance_m above the surface.
            floor_lift_m = float(self.lift_position) + (surface_z + self.finger_clearance_m - gripper_bottom_z)
            # Where to stop stepping down: the object pressed press_m into the surface.
            stop_lift_m = float(self.lift_position) + (surface_z - self.press_m - object_bottom_z)
            stop_lift_m = max(stop_lift_m, floor_lift_m, min_lift_m)

            # 2. Lower quickly until the object is approach_gap_m above the surface.
            gap_m = self.approach_gap_m if object_seen else self.unseen_object_gap_m
            approach_lift_m = float(self.lift_position) + (surface_z + gap_m - object_bottom_z)
            approach_lift_m = min(max(approach_lift_m, stop_lift_m), max_lift_m)
            baseline = None
            if approach_lift_m < float(self.lift_position) - 0.005:
                self.logger.info(f'Lower the object to {100 * gap_m:.0f} cm above the surface.')
                self.debug_move_to_pose({'joint_lift': approach_lift_m})
                # Effort at rest after moving down, to compare each step down against. Without
                # this move, the first step sets it, so the lift has moved the same way.
                baseline = self._lift_effort()

            # 3. Step down until the surface takes the object's weight.
            how = 'reached the lowest safe height'
            lift_m = float(self.lift_position)
            while lift_m - self.step_m >= stop_lift_m - 1e-4:
                lift_m -= self.step_m
                self.debug_move_to_pose({'joint_lift': lift_m})
                effort = self._lift_effort()
                if effort is None:
                    continue
                if baseline is None:
                    baseline = effort
                    continue
                if baseline - effort >= self.contact_effort_drop_pct:
                    how = f'felt contact (lift effort {baseline:.1f} -> {effort:.1f} %)'
                    break
            self.logger.info(f'Stopped lowering: {how}.')

            # 4. Release and lift clear.
            self.logger.info('Open the gripper to release the object.')
            self.debug_move_to_pose({'gripper_aperture': float(self.release_aperture_m)})
            time.sleep(self.release_wait_s)
            if self.lift_position is not None:
                self.logger.info('Lift the gripper clear of the object.')
                self.debug_move_to_pose({'joint_lift': min(float(self.lift_position) + self.raise_after_m, max_lift_m)})

            response.success = True
            response.message = (f'Placed the object on the surface at {surface_z:.3f} m: {how}; '
                                f'released and lifted {100 * self.raise_after_m:.0f} cm.')
            self.logger.info(response.message)
            return response
        except DebugMoveCancelled as exc:
            return self._fail(response, str(exc))

    def main(self):
        hm.HelloNode.main(self, 'place_object', 'place_object', wait_for_first_pointcloud=False)
        self.logger = self.get_logger()
        self.callback_group = ReentrantCallbackGroup()

        defaults = {
            'place_service': '/place_object',
            'dryrun': False,
            'debug_mode': False,
            'approach_gap_m': 0.03,
            'unseen_object_gap_m': 0.20,
            'step_m': 0.01,
            'contact_effort_drop_pct': 8.0,
            'press_m': 0.02,
            'finger_clearance_m': 0.01,
            'release_aperture_m': 0.07,
            'release_wait_s': 1.0,
            'raise_after_m': 0.05,
            'head_settle_s': 1.0,
        }
        for name, default in defaults.items():
            self._declare_parameter_once(name, default)
        place_service = self.get_parameter('place_service').value
        self.dryrun = bool(self.get_parameter('dryrun').value)
        self.debug_mode = bool(self.get_parameter('debug_mode').value)
        for name in ('approach_gap_m', 'unseen_object_gap_m', 'step_m', 'contact_effort_drop_pct', 'press_m', 'finger_clearance_m',
                     'release_aperture_m', 'release_wait_s', 'raise_after_m', 'head_settle_s'):
            setattr(self, name, float(self.get_parameter(name).value))

        self.joint_states_subscriber = self.create_subscription(
            JointState, '/stretch/joint_states', callback=self.joint_states_callback,
            qos_profile=1, callback_group=self.callback_group)
        self.place_service = self.create_service(
            Trigger, place_service, self.place_object_service_callback, callback_group=self.callback_group)
        self.debug_marker_pub = self.create_publisher(Marker, '/place_object/debug_translation_marker', 1)
        self.logger.info(f'Place object node is ready. Service: {place_service}')


def main():
    try:
        node = PlaceObjectNode()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        rclpy.logging.get_logger('place_object').info('interrupt received, so shutting down')


if __name__ == '__main__':
    main()
