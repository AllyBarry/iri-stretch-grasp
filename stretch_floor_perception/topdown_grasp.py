#!/usr/bin/env python3

import math
import time

import numpy as np
import rclpy
import rclpy.logging
from geometry_msgs.msg import Point, Quaternion
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

import hello_helpers.hello_misc as hm
from stretch_floor_perception.grasp_carry import DebugMoveCancelled, GraspPointNode
from stretch_floor_perception.srv import GraspPoint


class TopdownGraspNode(GraspPointNode):
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
        self.max_center_shift_fraction = 0.5
        self.min_max_center_shift_m = 0.03
        self.roll_axis_offset_rad = 0.0
        self.topdown_wrist_pitch_rad = (-math.pi / 2.0) + 0.07
        self.topdown_wrist_yaw_rad = 0.0
        self.min_pitch_clearance_m = 0.05
        self.align_tolerance_m = 0.005
        self.open_aperture_m = 0.075
        self.object_lift_height_m = 0.10
        self.retract_after_lift = True

    # ------------------------------------------------------------------
    # Perception helpers (measuring the object is shared, in grasp_carry)
    # ------------------------------------------------------------------

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

        The inverse of _compute_topdown_roll: the fingers close along -roll, less the offset.
        """
        angle = -(float(roll) - float(self.roll_axis_offset_rad))
        return np.array([math.cos(angle), math.sin(angle)])

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

    def _publish_actual_grip(self, roll, center_base=None):
        """Show where the fingertips really are (magenta) and log how far their closing line is
        from the planned one, which is the correction for roll_axis_offset_rad, and how far
        their midpoint is from the planned centre."""
        left = self._get_frame_in_base('link_gripper_fingertip_left')
        right = self._get_frame_in_base('link_gripper_fingertip_right')
        if left is None or right is None:
            return
        self._publish_lines('topdown_grip_actual', [left, right], (1.0, 0.1, 0.9), 0.006, with_ends=True)
        if center_base is not None:
            offset = 0.5 * (left + right)[:2] - np.asarray(center_base, dtype=float)[:2]
            self.logger.info(
                f'Fingertip midpoint is {100 * float(np.linalg.norm(offset)):.1f} cm from the planned centre '
                f'(base x {100 * offset[0]:+.1f} cm, y {100 * offset[1]:+.1f} cm).'
            )
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
            f'(base_link), off by {error_deg:+.0f} deg. If this stays off, add '
            f'{math.radians(error_deg):+.3f} rad to roll_axis_offset_rad.'
        )

    def _clear_topdown_markers(self):
        """Remove the planned object and grip markers. Kept apart from _clear_translation_marker,
        which the debug prompt calls before every move."""
        self._delete_markers((('topdown_object', 0), ('topdown_long_axis', 0),
                              ('topdown_grip_planned', 0), ('topdown_grip_planned', 1),
                              ('topdown_grip_actual', 0), ('topdown_grip_actual', 1), ('table_edge', 0)))

    def _compute_topdown_roll(self, major_axis_xy):
        """Wrist roll that makes the fingers close across the object's long axis.

        With the gripper pointing down, the fingers close along base x at roll 0, and a
        positive roll turns that line clockwise seen from above, so they close along
        -roll (checked against the URDF and on the robot). Closing across the long axis
        therefore needs roll = -(long axis angle + 90 deg). roll_axis_offset_rad fine-tunes it.
        """
        axis_xy = np.asarray(major_axis_xy, dtype=float)
        closing = math.atan2(float(axis_xy[1]), float(axis_xy[0])) + (math.pi / 2.0)
        roll = -closing + float(self.roll_axis_offset_rad)
        roll = math.atan2(math.sin(roll), math.cos(roll))

        # Top-down finger alignment is symmetric under pi rotation, so choose
        # the nearest equivalent angle within the typical wrist roll range.
        if roll > (math.pi / 2.0):
            roll -= math.pi
        elif roll < (-math.pi / 2.0):
            roll += math.pi
        return roll

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

    def topdown_grasp_service_callback(self, request, response):
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

            # Remember where the object was, in odom, to check afterwards that it is gone. The
            # base turns and drives below, so the target and box are followed in odom too.
            pick_odom = self._transform_point(
                [msg.point.x, msg.point.y, msg.point.z], msg.header.frame_id, 'odom')
            if pick_odom is None:
                return self._fail(response, 'Unable to transform the target point into odom.')
            box_odom = None
            box_request = self._bbox_in_base(request.bbox, msg.header.frame_id)
            if box_request is not None:
                box_odom = self._box_to_frame(box_request, 'base_link', 'odom')

            # 1. Retract the arm at its current height.
            self.logger.info('Retract the arm before planning.')
            self.debug_move_to_pose({'wrist_extension': min_extension_m})

            # 2. Square the base to the table edge found in the surface scan, so the arm reaches
            #    straight out over the table and the base later drives along the edge.
            if self.square_to_table_edge:
                manip = self.look_at_surface(scan_time_s=self.scan_time_s)
                edge = self._detect_table_edge(manip, pick_odom) if manip is not None else None
                square_turn_rad = self._square_turn(edge) if edge is not None else 0.0
                if abs(square_turn_rad) > 1e-3:
                    self.logger.info(f'Turn the base {math.degrees(square_turn_rad):+.0f} deg to square up to the table edge.')
                    self.debug_move_to_pose({'rotate_mobile_base': square_turn_rad})

            # 3. Look at the target and measure the object.
            target_base = self._transform_point(pick_odom, 'odom', 'base_link')
            if target_base is None:
                return self._fail(response, 'Unable to transform the target point into base_link.')
            self._point_head_at(target_base, 'the target')
            time.sleep(self.head_settle_s)
            point_cloud_msg = self._wait_for_fresh_point_cloud()
            if point_cloud_msg is None:
                return self._fail(response, 'No fresh point cloud available, so unable to plan the top-down grasp.')

            box = self._box_to_frame(box_odom, 'odom', 'base_link') if box_odom is not None else None
            if box is None:
                self.logger.warn('No detector box in the request, so measuring the object from the target alone.')
            else:
                self.logger.info(
                    f'Detector box: center=({box["center"][0]:.3f}, {box["center"][1]:.3f}) '
                    f'size=({box["size"][0]:.3f}, {box["size"][1]:.3f}, {box["size"][2]:.3f}) m '
                    f'axis={math.degrees(math.atan2(box["major_axis_xy"][1], box["major_axis_xy"][0])):.0f} deg'
                )
            crop_radius_m = self._crop_radius(box)

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

            # 4. Drive the base in line with the object while the arm is still retracted at its
            #    current height. Squared up, the base drives along the table edge. The arm's line
            #    is the wrist yaw axis, which the gripper will point straight out from.
            target_base = object_in_base()
            yaw_axis_base = self._get_frame_in_base('link_wrist_yaw')
            if target_base is None or yaw_axis_base is None:
                return self._fail(response, 'Unable to locate the object or the wrist.')
            self.logger.info('Drive in line with the object.')
            self.debug_drive(float(target_base[0] - yaw_axis_base[0]))

            # 5. Look again: odometry drifts while turning and driving, so measure the object
            #    from where the base now is. The arm is still low, so it does not block the view.
            if self.reobserve_after_base_move:
                again = self._reobserve_object(object_odom, box_odom, assumed_width_m)
                if again is not None:
                    corrected_odom = self._transform_point(
                        np.array([again['center_xy'][0], again['center_xy'][1], obj['top_z']]), 'base_link', 'odom')
                    if corrected_odom is not None:
                        object_odom = corrected_odom
                        obj['center_xy'] = again['center_xy']
                        self._publish_object_marker(obj)
                        self._publish_planned_grip(obj, grasp_axis, topdown_roll)

            # Distance from the wrist pitch axis to the gripper tip. Tilting the
            # gripper down drops the tip by about this much below the pitch axis.
            tip_base = self._get_gripper_tip_in_base()
            pitch_base = self._get_frame_in_base('link_wrist_pitch')
            if tip_base is None or pitch_base is None:
                return self._fail(response, 'Unable to locate the wrist and gripper tip.')
            tip_drop_m = float(np.linalg.norm(tip_base - pitch_base)) * math.cos(
                float(self.topdown_wrist_pitch_rad) + (math.pi / 2.0))

            # 6. Raise the retracted arm so that, once tilted down over the object,
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

            # 7. Point the gripper level and straight out along the arm, ready to extend.
            self.logger.info('Level the gripper, pointing out along the arm.')
            self.debug_move_to_pose({
                'joint_wrist_pitch': 0.0,
                'joint_wrist_yaw': float(self.topdown_wrist_yaw_rad),
                'joint_wrist_roll': 0.0,
            })

            # 8. Extend from above so the wrist pitch axis is over the object.
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

            # 9. Over the object: tilt the gripper down and align the fingers.
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
            self._publish_actual_grip(topdown_roll, object_in_base())

            # 10. Continue straight down to surround the object, close and lift straight up.
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

            # 11. Present the gripper to the camera and confirm the object is held.
            # The grasp verifier moves to its own check pose; otherwise show the gripper here.
            use_verifier = self.verify_grasp_visually and (not self.dryrun) and self._verifier_ready()
            if not use_verifier:
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

            response.success, response.message = self._verify_grasp(
                request.expected_label, pick_odom, use_verifier)
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
        hm.HelloNode.main(self, 'topdown_grasp', 'topdown_grasp', wait_for_first_pointcloud=False)
        self.logger = self.get_logger()

        self.callback_group = ReentrantCallbackGroup()

        defaults = {
            'debug_directory': '',
            'assumed_width_m': 0.05,
            'grasp_service': '/topdown_grasp',
            'dryrun': False,
            'debug_mode': False,
            # Carry pose and visual confirmation (shared with grasp_carry).
            'carry_wrist_pitch_rad': -0.02,
            'carry_wrist_yaw_rad': 0.70,
            # After a top-down grasp the wrist roll can be anywhere, so take the nearer way up.
            'carry_wrist_roll_nearest': True,
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
            # Squaring up to the table edge, and looking again after the base moves.
            'square_to_table_edge': True,
            'max_square_turn_rad': 0.52,
            'reobserve_after_base_move': True,
            'max_reobserve_shift_m': 0.05,
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
        self.square_to_table_edge = bool(self.get_parameter('square_to_table_edge').value)
        self.reobserve_after_base_move = bool(self.get_parameter('reobserve_after_base_move').value)
        self.min_object_points = int(self.get_parameter('min_object_points').value)
        self.min_cluster_points = int(self.get_parameter('min_cluster_points').value)
        for name in (
            'assumed_width_m', 'carry_wrist_pitch_rad', 'carry_wrist_yaw_rad',
            'object_cluster_radius_m', 'object_vertical_window_m', 'min_object_height_above_surface_m',
            'max_object_height_m', 'min_elongation_ratio', 'roll_axis_offset_rad',
            'bbox_margin_m', 'cluster_cell_m', 'max_center_shift_fraction', 'min_max_center_shift_m',
            'top_band_fraction', 'min_top_band_m', 'max_square_turn_rad', 'max_reobserve_shift_m',
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
            GraspPoint, grasp_service, self.topdown_grasp_service_callback,
            callback_group=self.callback_group)

        self.debug_marker_pub = self.create_publisher(
            Marker, '/topdown_grasp/debug_translation_marker', 1)

        self.logger.info(f'Top-down grasp point node is ready. Service: {grasp_service}')


def main():
    try:
        node = TopdownGraspNode()
        node.main()
        node.new_thread.join()
    except KeyboardInterrupt:
        rclpy.logging.get_logger('topdown_grasp').info('interrupt received, so shutting down')


if __name__ == '__main__':
    main()
