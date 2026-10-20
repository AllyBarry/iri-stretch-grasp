#!/usr/bin/env python3
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import CameraInfo, CompressedImage

from stretch_floor_perception.srv import DetectFloorObjects3D
from stretch_floor_perception.srv import GraspPoint


class ManualScanClient(Node):
    def __init__(self):
        super().__init__("scan_client")

        self.declare_parameter("color_topic", "/camera/color/image_raw/compressed")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw/compressedDepth")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("service_name", "detect_floor_objects_3d")
        self.declare_parameter("grasp_service_name", "/grasp_carry")
        self.declare_parameter("topdown_grasp_service_name", "/topdown_grasp")

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self._lock = threading.Lock()
        self._latest_color = None
        self._latest_depth = None
        self._latest_info = None

        self.create_subscription(
            CompressedImage, self.get_parameter("color_topic").value, self._on_color, qos
        )
        self.create_subscription(
            CompressedImage, self.get_parameter("depth_topic").value, self._on_depth, qos
        )
        self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value, self._on_info, qos
        )

        service_name = self.get_parameter("service_name").value
        self._client = self.create_client(DetectFloorObjects3D, service_name)
        self.get_logger().info(f"Waiting for service '{service_name}'...")
        self._client.wait_for_service()
        self.get_logger().info("Service available.")

        grasp_service_name = self.get_parameter("grasp_service_name").value
        self._grasp_client = self.create_client(GraspPoint, grasp_service_name)
        # Testing the top-down grasp: don't block on the front grasp service.
        # self.get_logger().info(f"Waiting for service '{grasp_service_name}'...")
        # self._grasp_client.wait_for_service()
        # self.get_logger().info("Grasp service available.")

        topdown_grasp_service_name = self.get_parameter("topdown_grasp_service_name").value
        self._topdown_grasp_client = self.create_client(GraspPoint, topdown_grasp_service_name)
        if self._topdown_grasp_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().info(
                f"Top-down grasp service '{topdown_grasp_service_name}' is available."
            )
        else:
            self.get_logger().warning(
                f"Top-down grasp service '{topdown_grasp_service_name}' is not available yet."
            )

    def _on_color(self, msg: CompressedImage):
        with self._lock:
            self._latest_color = msg

    def _on_depth(self, msg: CompressedImage):
        with self._lock:
            self._latest_depth = msg

    def _on_info(self, msg: CameraInfo):
        with self._lock:
            self._latest_info = msg

    def run_scan(self) -> None:
        with self._lock:
            color_msg = self._latest_color
            depth_msg = self._latest_depth
            info_msg = self._latest_info

        if color_msg is None or depth_msg is None or info_msg is None:
            print("Scan skipped: waiting for latest color, depth, and camera_info messages.")
            return

        request = DetectFloorObjects3D.Request()
        request.color_msg = color_msg
        request.depth_msg = depth_msg
        request.info_msg = info_msg

        future = self._client.call_async(request)
        while rclpy.ok() and not future.done():
            time.sleep(0.05)

        if not future.done():
            print("Scan canceled before service response.")
            return

        exception = future.exception()
        if exception is not None:
            print(f"Service call failed: {exception}")
            return

        response = future.result()
        detections = response.detections.detections
        if not detections:
            print("No objects detected.")
            return

        print(f"Detected {len(detections)} object(s):")
        for idx, det in enumerate(detections):
            p = det.centroid
            print(
                f"{idx}: object='{det.class_name}' detector='{det.detector_class_name}' "
                f"pick_up='{det.pick_up}' grasp='{det.grasp_direction}' score={det.score:.3f} "
                f"xyz=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
            )

        selection = self._prompt_for_detection_index(len(detections))
        if selection is None:
            print("No object selected.")
            return

        selected = detections[selection]
        self._call_grasp_carry(selected)

    def _prompt_for_detection_index(self, num_detections: int):
        prompt = f"Select object index to grasp [0-{num_detections - 1}] (or 'q' to cancel): "
        while rclpy.ok():
            user_input = input(prompt).strip().lower()
            if user_input == "q":
                return None
            try:
                index = int(user_input)
            except ValueError:
                print("Invalid selection. Enter a numeric index or 'q'.")
                continue

            if 0 <= index < num_detections:
                return index

            print(f"Index out of range. Choose a value from 0 to {num_detections - 1}.")
        return None

    def _call_grasp_carry(self, detection) -> None:
        grasp_direction = str(detection.grasp_direction).strip().lower()
        # Testing the top-down grasp: always use it, whatever grasp_direction says.
        # if grasp_direction == "top":
        #     client = self._topdown_grasp_client
        #     service_name = self.get_parameter("topdown_grasp_service_name").value
        # else:
        # client = self._grasp_client
        # service_name = self.get_parameter("grasp_service_name").value
        client = self._topdown_grasp_client
        service_name = self.get_parameter("topdown_grasp_service_name").value

        if not client.wait_for_service(timeout_sec=2.0):
            print(f"Selected grasp service '{service_name}' is unavailable.")
            return

        request = GraspPoint.Request()
        request.target.header = detection.header
        request.target.point = detection.centroid
        # The box lets the grasp node keep to this object's points and align with its long axis.
        request.bbox = detection.bbox3d
        # The grasp node confirms the grasp by detecting this label in the gripper. Use the
        # detector's own label, since that is what it will detect (not the VLM's name).
        request.expected_label = str(detection.detector_class_name or detection.class_name)

        print(
            f"Calling grasp service '{service_name}' for '{request.expected_label}', "
            f"grasp_direction='{grasp_direction or 'unsure'}'."
        )
        future = client.call_async(request)
        while rclpy.ok() and not future.done():
            time.sleep(0.05)

        if not future.done():
            print("Grasp request canceled before service response.")
            return

        exception = future.exception()
        if exception is not None:
            print(f"Grasp service call failed: {exception}")
            return

        response = future.result()
        print(f"Grasp response: success={response.success} message='{response.message}'")


def main():
    rclpy.init()
    node = ManualScanClient()

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            user_input = input("Press Enter to scan, or 'q' then Enter to quit: ").strip().lower()
            if user_input == "q":
                break
            node.run_scan()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
