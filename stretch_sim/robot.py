from dataclasses import dataclass, field
from typing import Optional

from stretch_sim.geometry import Pose2D, local_to_world

DRIVER_MODES = ("position", "navigation", "trajectory")


@dataclass
class RobotGeometry:
    '''
    Simplified Stretch kinematics. The arm extends towards the base's -y (right) side;
    reach is measured from the base origin to the gripper centre.
    '''
    base_size: float = 0.34
    radius: float = 0.22               # for path planning
    arm_min_reach: float = 0.25        # gripper centre, arm fully retracted
    arm_max_extension: float = 0.52
    lift_min: float = 0.2
    lift_max: float = 1.1
    gripper_open_m: float = 0.10
    finger_length_m: float = 0.171     # matches the aperture conversion used by the BT
    max_align_m: float = 0.8           # how far a grasp may shift the base along its heading

    @property
    def arm_max_reach(self) -> float:
        return self.arm_min_reach + self.arm_max_extension


@dataclass
class RobotState:
    pose: Pose2D = field(default_factory=Pose2D)
    driver_mode: str = "position"
    head_pan: float = 0.0
    head_tilt: float = 0.0
    lift: float = 0.6
    arm_extension: float = 0.0
    aperture: float = 0.0              # gripper opening in metres
    held: Optional[str] = None

    def copy(self) -> "RobotState":
        return RobotState(self.pose.copy(), self.driver_mode, self.head_pan, self.head_tilt,
                          self.lift, self.arm_extension, self.aperture, self.held)

    def gripper_xy(self, geom: RobotGeometry):
        return local_to_world(self.pose, 0.0, -(geom.arm_min_reach + self.arm_extension))

    def finger_joint_rad(self, geom: RobotGeometry) -> float:
        '''joint_gripper_finger_left for this aperture (inverse of aperture = 2 * rad * finger_length).'''
        return self.aperture / (2.0 * geom.finger_length_m)
