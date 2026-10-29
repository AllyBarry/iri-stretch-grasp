#!/usr/bin/env python3
"""carry_pose.py: a /carry_pose service that puts Stretch in a pose for carrying an object.

Run on the robot with the driver running, then call the service whenever needed:

  ros2 run stretch_floor_perception carry_pose.py
  ros2 run stretch_floor_perception carry_pose.py --ros-args -p dryrun:=true   # log moves only
  ros2 service call /carry_pose std_srvs/srv/Trigger

The arm retracts first, then the gripper swings toward the front of the base and turns so the
held object is vertical. At roll 0 (or +-pi) the fingers close horizontally, so an object gripped
from the side stands upright; the roll turns to whichever of those is nearer, never the long way
round. Pitch goes to level, or stays pointing down after a top-down grasp, where the object is
already vertical and roll only spins it. The lift and gripper are not commanded.
"""
import math
import threading

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

CARRY_WRIST_YAW = 1.57          # gripper toward the front of the base
PITCH_RANGE = (-1.56, 0.0)      # level or pointing straight down, inside the driver's limits
ROLL_RANGE = (-2.9, 2.9)        # roll limit is +-2.91, so +-pi is reached only to within 14 deg


def snap(x, step):
    return round(x / step) * step


def clamp(x, lo_hi):
    return min(max(x, lo_hi[0]), lo_hi[1])


class CarryPose(Node):
    def __init__(self):
        super().__init__('carry_pose')
        self.dryrun = bool(self.declare_parameter('dryrun', False).value)
        group = ReentrantCallbackGroup()
        self.joint_states = None
        self.joint_states_lock = threading.Lock()
        self.create_subscription(JointState, '/stretch/joint_states', self._joint_states_cb, 1,
                                 callback_group=group)
        self.position_mode = self.create_client(Trigger, '/switch_to_position_mode', callback_group=group)
        self.trajectory = ActionClient(self, FollowJointTrajectory,
                                       '/stretch_controller/follow_joint_trajectory', callback_group=group)
        self.create_service(Trigger, '/carry_pose', self._carry_cb, callback_group=group)
        self.get_logger().info('Ready: ros2 service call /carry_pose std_srvs/srv/Trigger')

    def _joint_states_cb(self, msg):
        with self.joint_states_lock:
            self.joint_states = dict(zip(msg.name, msg.position))

    def _wait(self, future, timeout_s):
        done = threading.Event()
        future.add_done_callback(lambda _: done.set())
        return future.result() if done.wait(timeout_s) else None

    def _move(self, positions, sec):
        """Send one trajectory point and wait for it; return an error string, or None on success."""
        if self.dryrun:
            self.get_logger().info('dryrun: skip move to {}'.format(positions))
            return None
        if not self.trajectory.wait_for_server(timeout_sec=5.0):
            return 'follow_joint_trajectory action server is not available'
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(positions)
        goal.trajectory.points = [JointTrajectoryPoint(positions=[float(p) for p in positions.values()],
                                                       time_from_start=Duration(sec=sec))]
        handle = self._wait(self.trajectory.send_goal_async(goal), 5.0)
        if handle is None or not handle.accepted:
            return 'goal {} was rejected'.format(positions)
        result = self._wait(handle.get_result_async(), sec + 15.0)
        if result is None:
            return 'goal {} timed out'.format(positions)
        if result.result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            return 'goal {} failed: {}'.format(positions, result.result.error_string)
        return None

    def _carry_cb(self, request, response):
        response.success = False
        if self.position_mode.wait_for_service(timeout_sec=5.0):
            self._wait(self.position_mode.call_async(Trigger.Request()), 5.0)

        self.get_logger().info('Retract the arm.')
        err = self._move({'joint_arm': 0.0}, 4)
        if err:
            response.message = err
            self.get_logger().error(err)
            return response

        with self.joint_states_lock:
            joints = self.joint_states
        if joints is None or 'joint_wrist_pitch' not in joints:
            response.message = 'no wrist joint states received on /stretch/joint_states'
            self.get_logger().error(response.message)
            return response

        pitch = clamp(snap(joints['joint_wrist_pitch'], math.pi / 2), PITCH_RANGE)
        roll = joints['joint_wrist_roll']
        if pitch > -0.5:  # gripper level: fingers must close horizontally for the object to stand upright
            roll = clamp(snap(roll, math.pi), ROLL_RANGE)

        self.get_logger().info('Tuck the gripper: pitch {:.2f}, roll {:.2f}.'.format(pitch, roll))
        err = self._move({'joint_wrist_yaw': CARRY_WRIST_YAW, 'joint_wrist_pitch': pitch,
                          'joint_wrist_roll': roll}, 3)
        if err:
            response.message = err
            self.get_logger().error(err)
            return response

        response.success = True
        response.message = 'carry pose reached (pitch {:.2f}, roll {:.2f})'.format(pitch, roll)
        return response


def main():
    rclpy.init()
    node = CarryPose()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
