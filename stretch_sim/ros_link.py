'''
The simulator's ROS side and which robot the behaviour tree drives, switchable at run time
(the viewer's Connect / Disconnect):

    sim      the sim plays the robot: it offers the robot's interfaces (ros_bridge.py),
             on this machine only unless --ros-network
    robot    the BT drives the real Stretch and the sim mirrors it (robot_link.py): the
             sim offers nothing, and joins the robot's ROS domain
    None     no ROS (started without --ros, --bt or --robot)

The two never overlap: switching stops the running BT and shuts the old ROS context down
before the new one starts, so a BT never sees the sim's services alongside the robot's.
ROS settings are fixed per context, so each mode gets a fresh one. The BT is a separate
process and takes the ROS environment it is started with.
'''
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

from stretch_sim.bt_runner import BtProcess, isolate_ros_discovery
from stretch_sim.events import INFO
from stretch_sim.robot_link import (DEFAULT_DOMAIN_ID, DEFAULT_PLACES, RobotMirror, location_poses_from_places,
                                    read_places, write_location_poses)
from stretch_sim.simulator import KinematicSimulator

_ROS_ENV = ("ROS_DOMAIN_ID", "ROS_AUTOMATIC_DISCOVERY_RANGE", "ROS_LOCALHOST_ONLY")


class RosLink:
    def __init__(self, sim: KinematicSimulator, run_dir: Path, bt_command: Optional[List[str]] = None,
                 ros_network: bool = False, places: Optional[str] = None,
                 place_names: Optional[Dict[str, str]] = None, domain_id: Optional[int] = None,
                 yaw_offsets_deg: Optional[Dict[str, float]] = None):
        self.sim = sim
        self.run_dir = Path(run_dir)
        self.ros_network = ros_network
        self.places = places or DEFAULT_PLACES      # defaults for connecting to the robot
        self.place_names = place_names or {}
        self.yaw_offsets_deg = yaw_offsets_deg or {}
        self.domain_id = domain_id
        self.bt = BtProcess(sim, run_dir, bt_command)
        self.mode: Optional[str] = None
        self.runner = None
        self.robot: Optional[dict] = None           # the robot connection, in robot mode
        self.error: Optional[str] = None            # why ROS is unavailable, if it is
        self._env = {k: os.environ.get(k) for k in _ROS_ENV}
        self._sim_success_rate = sim.success_rate
        self._lock = threading.Lock()               # one switch at a time

    @property
    def bridge(self):
        return self.runner.bridge if self.runner is not None else None

    def status(self) -> dict:
        '''What the viewer shows about the ROS side; the shape run_sim used before switching existed.'''
        status = {"available": self.mode is not None, "mode": self.mode,
                  "connect": {"places": self.places, "domain_id": self.domain_id or self._default_domain()}}
        if self.mode is None:
            status["reason"] = self.error or "started without --ros"
        else:
            status["offered"], status["missing"] = list(self.bridge.offered), list(self.bridge.missing)
        if self.robot is not None:
            status["robot"] = self.robot
            status["runstopped"] = getattr(self.bridge, "runstopped", None)   # None until the robot reports it
        return status

    '''
    SWITCHING
    '''
    def start_sim(self) -> None:
        '''The sim plays the robot.'''
        with self._lock:
            self._stop()
            self._restore_env()
            if not self.ros_network:
                isolate_ros_discovery()
            self.sim.set_success_rate(self._sim_success_rate)
            self.bt.location_poses = None
            self.robot = None
            self._start_runner({})
            self.mode = "sim"
            self.sim.emit(INFO, "ros", "the sim now plays the robot")

    def start_robot(self, places: Optional[str] = None, domain_id: Optional[int] = None) -> None:
        '''
        Drive the real robot and mirror it. The places are read before anything is
        stopped, so a robot that cannot be reached leaves the current mode running.
        Raises RuntimeError (unreachable places file) or ImportError (no ROS).
        '''
        with self._lock:
            places = places or self.places
            domain = domain_id if domain_id is not None else self._default_domain()
            found = read_places(places)
            poses, missing = location_poses_from_places(found, list(self.sim.world.locations), self.place_names,
                                                        self.yaw_offsets_deg)
            path = write_location_poses(poses, self.run_dir / "robot_location_poses.json")
            self.places, self.domain_id = places, domain

            self._stop()
            self._restore_env()
            os.environ["ROS_DOMAIN_ID"] = str(domain)
            if (os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") or "").upper() in ("LOCALHOST", "OFF"):
                os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "SUBNET"
            if os.environ.get("ROS_LOCALHOST_ONLY") == "1":
                os.environ["ROS_LOCALHOST_ONLY"] = "0"
            if self.mode != "robot":
                self._sim_success_rate = self.sim.success_rate
            self.sim.set_success_rate(1.0)      # the mirror shows what the real robot did, never a simulated failure
            self.bt.location_poses = poses
            self.robot = {"places": places, "poses": poses, "missing": missing, "path": str(path), "domain_id": domain}
            # The real robot offers the interfaces; the sim only follows the BT.
            self._start_runner({"serve_robot": False, "on_bt_state": RobotMirror(self.sim).on_bt_state})
            self.mode = "robot"
            self.sim.emit(INFO, "ros", f"connected to the real robot on ROS domain {domain}; the map mirrors it",
                          places=places, poses=poses, missing=missing)
            if missing:
                self.sim.emit(INFO, "ros", f"no place on the robot for {', '.join(missing)}: the BT cannot navigate there",
                              places=sorted(found))

    def stop(self) -> None:
        with self._lock:
            self._stop()

    '''
    HELPERS
    '''
    def _start_runner(self, bridge_args: dict) -> None:
        from stretch_sim.ros_bridge import RosBridgeRunner     # imports rclpy
        domain = os.environ.get("ROS_DOMAIN_ID")
        self.runner = RosBridgeRunner(self.sim, domain_id=int(domain) if domain else None, **bridge_args)
        self.runner.start()
        self.error = None

    def _stop(self) -> None:
        self.bt.stop()
        if self.runner is not None:
            self.runner.stop()
            self.runner = None
        if self.mode is not None:
            self.sim.emit(INFO, "ros", "the sim no longer plays the robot" if self.mode == "sim"
                          else "disconnected from the real robot")
        self.mode = None
        self.robot = None

    def _restore_env(self) -> None:
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _default_domain(self) -> int:
        return self.domain_id if self.domain_id is not None else int(self._env.get("ROS_DOMAIN_ID") or DEFAULT_DOMAIN_ID)
