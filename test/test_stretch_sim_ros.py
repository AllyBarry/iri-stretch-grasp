'''
Tests for the simulator's ROS interface. They need rclpy, so run them with the ROS
Python in a sourced environment, e.g.:
    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 -m unittest test.test_stretch_sim_ros -v
Interfaces whose message packages are not installed are skipped.
'''
import os
import time
import unittest

os.environ.setdefault("ROS_AUTOMATIC_DISCOVERY_RANGE", "LOCALHOST")
os.environ.setdefault("ROS_DOMAIN_ID", "87")

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_srvs.srv import Trigger
except ImportError:          # not a ROS Python
    rclpy = None

from stretch_sim.config import load_scenario, resolve
from stretch_sim.world import load_world


@unittest.skipIf(rclpy is None, "rclpy not available")
class RosBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from stretch_sim.controller import SimController
        from stretch_sim.ros_bridge import RosBridgeRunner
        from stretch_sim.simulator import KinematicSimulator
        scenario = load_scenario("default") | {"perception": {"miss_prob": 0.0, "position_noise_m": 0.0}}
        cls.sim = KinematicSimulator(load_world(resolve("apartment", "worlds")), scenario)
        cls.controller = SimController(cls.sim, speed=8.0)
        cls.controller.start()
        cls.ros = RosBridgeRunner(cls.sim)
        cls.ros.start()
        if not rclpy.ok():
            rclpy.init()
        cls.client = Node("stretch_sim_test_client")

    @classmethod
    def tearDownClass(cls):
        cls.client.destroy_node()
        cls.controller.stop()
        cls.ros.stop()
        rclpy.try_shutdown()

    def _call(self, srv_type, name, request, timeout=20.0):
        client = self.client.create_client(srv_type, name)
        self.assertTrue(client.wait_for_service(timeout_sec=5.0), f"{name} not offered")
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self.client, future, timeout_sec=timeout)
        self.client.destroy_client(client)
        return future.result()

    def test_driver_mode_services(self):
        res = self._call(Trigger, "/switch_to_navigation_mode", Trigger.Request())
        self.assertTrue(res.success)
        self.assertEqual(self.sim.robot.driver_mode, "navigation")
        res = self._call(Trigger, "/switch_to_position_mode", Trigger.Request())
        self.assertTrue(res.success)
        self.assertEqual(self.sim.robot.driver_mode, "position")

    def test_handover_with_an_empty_gripper_fails(self):
        self._call(Trigger, "/switch_to_position_mode", Trigger.Request())
        res = self._call(Trigger, "/handover", Trigger.Request())
        self.assertFalse(res.success)
        self.assertIn("nothing in the gripper", res.message)

    def test_joint_states_report_the_gripper(self):
        received = []
        sub = self.client.create_subscription(JointState, "/stretch/joint_states", received.append, 10)
        end = time.monotonic() + 5.0
        while not received and time.monotonic() < end:
            rclpy.spin_once(self.client, timeout_sec=0.1)
        self.client.destroy_subscription(sub)
        self.assertTrue(received)
        self.assertIn("joint_gripper_finger_left", received[0].name)

    def test_published_bt_state_reaches_the_sim(self):
        import json
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        pub = self.client.create_publisher(String, "/cooking_bt/state", qos)
        payload = {"tree": "main_demo_loop", "status": "RUNNING", "tip": "SearchForObjectNode",
                   "values": {"target_object": "Salt", "confidence/Salt/Pantry": 0.8}}
        pub.publish(String(data=json.dumps(payload)))
        end = time.monotonic() + 5.0
        while self.sim.bt_state is None and time.monotonic() < end:
            rclpy.spin_once(self.client, timeout_sec=0.1)
        self.client.destroy_publisher(pub)
        self.assertEqual(self.sim.bt_state["tip"], "SearchForObjectNode")

    def test_navigation_action(self):
        try:
            from nav2_msgs.action import NavigateToPose
            from rclpy.action import ActionClient
        except ImportError:
            self.skipTest("nav2_msgs not installed")
        self._call(Trigger, "/switch_to_navigation_mode", Trigger.Request())
        client = ActionClient(self.client, NavigateToPose, "/navigate_to_pose")
        self.assertTrue(client.wait_for_server(timeout_sec=5.0))
        goal = NavigateToPose.Goal()
        a = self.sim.world.locations["Table"].approach
        goal.pose.header.frame_id = "map"
        goal.pose.pose.position.x, goal.pose.pose.position.y = a.x, a.y
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = __import__("math").sin(a.yaw / 2), __import__("math").cos(a.yaw / 2)
        send = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self.client, send, timeout_sec=5.0)
        result = send.result().get_result_async()
        rclpy.spin_until_future_complete(self.client, result, timeout_sec=60.0)
        self.assertEqual(result.result().status, 4)    # GoalStatus.STATUS_SUCCEEDED
        self.assertAlmostEqual(self.sim.robot.pose.x, a.x, places=2)

    def test_sim_offers_only_what_the_robot_offers(self):
        from stretch_sim.ros_bridge import ROBOT_INTERFACES
        self.assertLessEqual(set(self.ros.bridge.offered) - {"/camera/... (placeholder images)"}, ROBOT_INTERFACES)
        with self.assertRaises(ValueError):
            self.ros.bridge._service(Trigger, "/not_on_the_robot", lambda req, res: res)


@unittest.skipIf(rclpy is None, "rclpy not available")
class RosLinkTest(unittest.TestCase):
    '''Switching between the sim playing the robot and the real robot, as the viewer's Connect does.'''
    PLACES = "places:\n  table: {x: 1.0, y: 2.0, yaw: 0.5}\n  pantry: {x: 3.0, y: 4.0, yaw: 0.0}\n"
    ROBOT_DOMAIN = 88      # stands in for the robot's; nothing listens there

    def setUp(self):
        import tempfile
        from pathlib import Path
        from stretch_sim.controller import SimController
        from stretch_sim.ros_link import RosLink
        from stretch_sim.simulator import KinematicSimulator
        self.tmp = tempfile.TemporaryDirectory()
        places = Path(self.tmp.name) / "places.yaml"
        places.write_text(self.PLACES)
        self.sim = KinematicSimulator(load_world(resolve("apartment", "worlds")), load_scenario("default"))
        self.link = RosLink(self.sim, Path(self.tmp.name), places=str(places), domain_id=self.ROBOT_DOMAIN)
        self.controller = SimController(self.sim, link=self.link)
        self.env = {k: os.environ.get(k) for k in ("ROS_DOMAIN_ID", "ROS_AUTOMATIC_DISCOVERY_RANGE")}

    def tearDown(self):
        self.controller.stop()
        for k, v in self.env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def test_connect_and_disconnect(self):
        self.link.start_sim()
        self.assertIn("/navigate_to_pose", self.controller.ros_status["offered"])
        self.assertFalse(self.controller.robot_mode)

        message = self.controller.execute({"cmd": "connect"})
        self.assertIn(f"domain {self.ROBOT_DOMAIN}", message)
        self.assertTrue(self.controller.robot_mode)
        status = self.controller.ros_status
        self.assertEqual(status["offered"], [])                     # the robot offers everything
        self.assertEqual(status["robot"]["poses"]["Table"], {"x": 1.0, "y": 2.0, "yaw": 0.5})
        self.assertEqual(os.environ["ROS_DOMAIN_ID"], str(self.ROBOT_DOMAIN))   # what the BT is started with
        self.assertEqual(self.controller.bt.location_poses["Pantry"], {"x": 3.0, "y": 4.0, "yaw": 0.0})
        self.assertEqual(self.sim.success_rate, 1.0)

        self.controller.execute({"cmd": "disconnect"})
        self.assertFalse(self.controller.robot_mode)
        self.assertIn("/navigate_to_pose", self.controller.ros_status["offered"])
        self.assertEqual(os.environ.get("ROS_DOMAIN_ID"), self.env["ROS_DOMAIN_ID"])
        self.assertIsNone(self.controller.bt.location_poses)

    def test_unreachable_robot_leaves_the_sim_running(self):
        from stretch_sim.controller import CommandError
        self.link.start_sim()
        with self.assertRaises(CommandError):
            self.controller.execute({"cmd": "connect", "places": "/no/such/places.yaml"})
        self.assertEqual(self.link.mode, "sim")
        self.assertIn("/navigate_to_pose", self.controller.ros_status["offered"])


if __name__ == "__main__":
    unittest.main()
