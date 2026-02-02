#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time, Duration
from rclpy.executors import MultiThreadedExecutor

from geometry_msgs.msg import Point, Quaternion, Vector3
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection2DArray, BoundingBox3D
from cv_bridge import CvBridge
import numpy as np
import cv2
from tf_transformations import quaternion_matrix


from stretch_floor_perception.msg import DetectedObject3D, DetectedObject3DArray
from stretch_floor_perception.srv import DetectFloorObjects3D

try:
    from tf2_ros import Buffer, TransformListener
except ImportError:
    Buffer = None
    TransformListener = None


from stretch_floor_perception.inference_backends import InferenceBackend


class FloorObjectDetector(Node):
    def __init__(self):
        super().__init__("floor_object_detector")

        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("aligned_depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("yolo_backend", "yolo-darknet")
        self.declare_parameter("yolo_model_path", "")
        self.declare_parameter("yolo_config_file", "")
        self.declare_parameter("yolo_class_names_file", "")
        self.declare_parameter("publish_mask_topic", False)
        self.declare_parameter("mask_topic", "~/mask")

        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT

        self.color_sub = self.create_subscription(
            Image, self.get_parameter("color_topic").value, self._on_color, qos
        )
        self.depth_sub = self.create_subscription(
            Image, self.get_parameter("aligned_depth_topic").value, self._on_depth, qos
        )
        self.info_sub = self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value, self._on_info, qos
        )

        self.det2d_pub = self.create_publisher(Detection2DArray, "detections_2d", 10)
        self.image_with_boxes_pub = self.create_publisher(Image, "image_with_boxes", 10)
        self.det3d_pub = self.create_publisher(DetectedObject3DArray, "detections_3d", 10)
        self.centroid_pub = self.create_publisher(MarkerArray, "detected_centroids", 10)
        self.bbox3d_pub = self.create_publisher(MarkerArray, "detected_bboxes_3d", 10)
        self.mask_pub = None
        if self.get_parameter("publish_mask_topic").value:
            self.mask_pub = self.create_publisher(
                Image, self.get_parameter("mask_topic").value, 10
            )

        self._color_msg = None
        self._depth_msg = None
        self._info_msg = None
        self.img_bridge = CvBridge()

        self._tf_buffer = Buffer() if Buffer is not None else None
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True) if self._tf_buffer else None

        self.inference_backend = InferenceBackend(
            self.get_parameter("yolo_backend").value,
            cfg_file=self.get_parameter("yolo_config_file").value,
            names_file=self.get_parameter("yolo_class_names_file").value,
            weights_file=self.get_parameter("yolo_model_path").value,
            logger=self.get_logger(),
            verbose=True, # TODO make param
        )
        self.service = self.create_service(
            DetectFloorObjects3D, "detect_floor_objects_3d", self._handle_detect_service
        )

    def _on_color(self, msg: Image):
        self._color_msg = msg

    def _on_depth(self, msg: Image):
        self._depth_msg = msg

    def _on_info(self, msg: CameraInfo):
        self._info_msg = msg

    def _pad_to_center_square(self, image: np.ndarray) -> tuple[np.ndarray, int, int]:
        height, width = image.shape[:2]
        if height == width:
            return image, 0, 0
        if height < width:
            pad_total = width - height
            pad_top = pad_total // 2
            pad_bottom = pad_total - pad_top
            padded = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0
            )
            return padded, 0, pad_top
        pad_total = height - width
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        padded = cv2.copyMakeBorder(
            image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
        )
        return padded, pad_left, 0

    def _get_intrinsics(self, info_msg):
        fx = info_msg.k[0]
        fy = info_msg.k[4]
        cx = info_msg.k[2]
        cy = info_msg.k[5]
        if fx == 0.0 or fy == 0.0:
            self.get_logger().warning("Camera intrinsics not available for depth backprojection.")
            return None
        return fx, fy, cx, cy
    
    def _clipped_window(self, u, v, box_size_x, box_size_y, width, height):
        half_w = box_size_y * 0.5
        half_h = box_size_x * 0.5
    
        xmin = int(max(0, np.floor(u - half_w)))
        xmax = int(min(width, np.ceil(u + half_w)))
        ymin = int(max(0, np.floor(v - half_h)))
        ymax = int(min(height, np.ceil(v + half_h)))
        return xmin, xmax, ymin, ymax

    def _backproject_depth_crop(self, depth_im, xmin, xmax, ymin, ymax, fx, fy, cx, cy):
        depth_crop = depth_im[ymin:ymax, xmin:xmax]
        valid = np.isfinite(depth_crop) & (depth_crop > 0.0)
        if not np.any(valid):
            return None
    
        ys, xs = np.nonzero(valid)          # NOTE: only valid pixels
        z = depth_crop[ys, xs].astype(np.float32)
    
        u = xs.astype(np.float32) + xmin
        v = ys.astype(np.float32) + ymin
    
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return np.stack((x, y, z), axis=1)


    def _get_3dpts_in_window(self, px, py, box_size_x, box_size_y, depth_im, info_msg, max_points = 20000):
        height, width = depth_im.shape[:2]
        intrinsics = self._get_intrinsics(info_msg)
        if intrinsics is None:
            return None
        fx, fy, cx, cy = intrinsics

        cx_pix, cy_pix = py, (height - 1.0 - px) # Image was rotated, need to flip x-y
        xmin, xmax, ymin, ymax = self._clipped_window(cx_pix, cy_pix, box_size_x, box_size_y, width, height)
        if xmax <= xmin or ymax <= ymin:
            self.get_logger().debug("Skipping detection: invalid bounding box after clipping (zero area)")
            return None
        
        points_cam = self._backproject_depth_crop(depth_im, xmin, xmax, ymin, ymax, fx, fy, cx, cy)
        if points_cam is None:
            self.get_logger().debug("Skipping detection: no valid depth values in bounding box")
            return None
        
        if points_cam.shape[0] > max_points:
            step = int(np.ceil(points_cam.shape[0] / max_points))
            points_cam = points_cam[::step]
    
        return points_cam


    def _identity_quaternion(self) -> Quaternion:
        return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

    def _make_detectedobject3d_msg(self, header, class_id, score, centroid, size) -> DetectedObject3D:
        det3d = DetectedObject3D()
        det3d.header = header
        det3d.class_name = class_id
        det3d.score = score
        det3d.centroid = Point(x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2]))
        bbox = BoundingBox3D()
        bbox.center.position = det3d.centroid
        bbox.center.orientation = self._identity_quaternion()
        bbox.size = Vector3(x=float(size[0]), y=float(size[1]), z=float(size[2]))
        det3d.bbox3d = bbox
        return det3d

    def _make_marker(
        self,
        header,
        ns,
        marker_id,
        marker_type,
        position,
        orientation,
        scale,
        color_rgb,
        alpha,
        lifetime_msg,
    ) -> Marker:
        marker = Marker()
        marker.header = header
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.position = position
        marker.pose.orientation = orientation
        marker.scale = scale
        marker.color.r = color_rgb[0]
        marker.color.g = color_rgb[1]
        marker.color.b = color_rgb[2]
        marker.color.a = alpha
        marker.lifetime = lifetime_msg
        return marker

    def _empty_det3d_msg(self, stamp=None, frame_id=None) -> DetectedObject3DArray:
        det3d_msg = DetectedObject3DArray()
        det3d_msg.header = Header()
        if stamp is not None:
            det3d_msg.header.stamp = stamp
        if frame_id is not None:
            det3d_msg.header.frame_id = frame_id
        det3d_msg.detections = []
        return det3d_msg

    def _handle_detect_service(self, request, response):
        det3d_msg, det2d_msg, image_with_boxes, centroid_markers, bbox3d_markers = self._process_once()
        response.detections = det3d_msg
        if det2d_msg is not None:
            self.det2d_pub.publish(det2d_msg)
        if image_with_boxes is not None:
            image_with_boxes_msg = self.img_bridge.cv2_to_imgmsg(
                image_with_boxes, encoding="rgb8"
            )
            image_with_boxes_msg.header = det2d_msg.header
            self.image_with_boxes_pub.publish(image_with_boxes_msg)
        if det3d_msg is not None:
            self.det3d_pub.publish(det3d_msg)
        if centroid_markers is not None:
            self.centroid_pub.publish(centroid_markers)
        if bbox3d_markers is not None:
            self.bbox3d_pub.publish(bbox3d_markers)
        return response

    def _process_once(self):
        if self._color_msg is None or self._depth_msg is None or self._info_msg is None:
            self.get_logger().warning(
                "Waiting for color, depth, and camera info messages before processing."
            )
            empty = self._empty_det3d_msg()
            return empty, None, None, None, None

        color_msg = self._color_msg
        depth_msg = self._depth_msg
        info_msg = self._info_msg

        color_img = self.img_bridge.imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
        color_img = cv2.rotate(color_img, cv2.ROTATE_90_CLOCKWISE)
        color_img, pad_left, pad_top = self._pad_to_center_square(color_img)
        inference_result = self.inference_backend.infer(color_img)
        if isinstance(inference_result, tuple):
            det2d_msg, image_with_boxes = inference_result
        else:
            det2d_msg = inference_result
            image_with_boxes = None
        det2d_msg.header = self._color_msg.header

        # TODO: publish segmentation mask output when backend provides it.
        if self.mask_pub is not None:
            pass

        det3d_msg = DetectedObject3DArray()
        det3d_msg.header = Header()
        det3d_msg.header.stamp = self._color_msg.header.stamp
        det3d_msg.header.frame_id = self.get_parameter("base_frame").value

        depth_img = self.img_bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        if depth_img.dtype == np.uint16:
            depth_m = depth_img.astype(np.float32) * 0.001
        else:
            depth_m = depth_img.astype(np.float32, copy=False)

        camera_frame = depth_msg.header.frame_id or self.get_parameter("camera_frame").value
        base_frame = self.get_parameter("base_frame").value

        if self._tf_buffer is None:
            self.get_logger().warning("TF buffer unavailable; skipping 3D detections.")
            return self._empty_det3d_msg(det3d_msg.header.stamp, det3d_msg.header.frame_id), det2d_msg, image_with_boxes, None, None
        try:
            transform = self._tf_buffer.lookup_transform(
                base_frame, camera_frame, Time.from_msg(depth_msg.header.stamp), timeout=Duration(seconds=1.0)
            )
        except Exception as exc:
            self.get_logger().warning(f"TF lookup failed: {exc}")
            return self._empty_det3d_msg(det3d_msg.header.stamp, det3d_msg.header.frame_id), det2d_msg, image_with_boxes, None, None

        t = transform.transform.translation
        q = transform.transform.rotation
        rot = quaternion_matrix([q.x, q.y, q.z, q.w])
        trans = np.array([t.x, t.y, t.z], dtype=np.float32)

        det3d_msg.detections = []
        centroid_markers = MarkerArray()
        bbox3d_markers = MarkerArray()

        n=0
        for det in det2d_msg.detections:
            if not det.results:
                continue
            best = max(det.results, key=lambda r: r.hypothesis.score)
            class_id = best.hypothesis.class_id
            score = float(best.hypothesis.score)

            cx_pix = det.bbox.center.position.x - pad_left
            cy_pix = det.bbox.center.position.y - pad_top

            pts3d_window_cam = self._get_3dpts_in_window(cx_pix, 
                                                         cy_pix, 
                                                         det.bbox.size_x, 
                                                         det.bbox.size_y, 
                                                         depth_m, 
                                                         info_msg,
                                                         max_points = 20000)
            if pts3d_window_cam is None:
                continue

            points_base = pts3d_window_cam @ rot.T[:3, :3] + trans[:3]
            centroid = points_base.mean(axis=0)

            min_xyz = points_base.min(axis=0)
            max_xyz = points_base.max(axis=0)
            size = max_xyz - min_xyz

            det3d = self._make_detectedobject3d_msg(det3d_msg.header, class_id, score, centroid, size)
            det3d_msg.detections.append(det3d)
            marker_lifetime = Duration(seconds=0.2).to_msg()
            centroid_marker = self._make_marker(
                det3d_msg.header,
                "detected_centroids",
                n,
                Marker.SPHERE,
                det3d.centroid,
                self._identity_quaternion(),
                Vector3(x=0.05, y=0.05, z=0.05),
                (0.1, 0.9, 0.2),
                1.0,
                marker_lifetime,
            )
            centroid_markers.markers.append(centroid_marker)
            bbox_marker = self._make_marker(
                det3d_msg.header,
                "detected_bboxes_3d",
                n,
                Marker.CUBE,
                det3d.centroid,
                det3d.bbox3d.center.orientation,
                det3d.bbox3d.size,
                (0.2, 0.4, 1.0),
                0.5,
                marker_lifetime,
            )
            bbox3d_markers.markers.append(bbox_marker)
            n += 1
        return det3d_msg, det2d_msg, image_with_boxes, centroid_markers, bbox3d_markers


def main():
    rclpy.init()
    node = FloorObjectDetector()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
