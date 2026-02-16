#!/usr/bin/env python3
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import CameraInfo, CompressedImage

from stretch_floor_perception.srv import DetectFloorObjects3D


class ManualScanClient(Node):
    def __init__(self):
        super().__init__("scan_client")

        self.declare_parameter("color_topic", "/camera/color/image_raw/compressed")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw/compressedDepth")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("service_name", "detect_floor_objects_3d")

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
                f"{idx}: class='{det.class_name}' score={det.score:.3f} "
                f"xyz=({p.x:.3f}, {p.y:.3f}, {p.z:.3f})"
            )


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
