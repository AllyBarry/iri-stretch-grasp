#!/usr/bin/env python3
import base64
import json
import rclpy
from rclpy.node import Node
from rclpy.time import Time, Duration
from rclpy.executors import MultiThreadedExecutor
from time import time

from geometry_msgs.msg import Point, Quaternion, Vector3
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import BoundingBox3D, Detection2D as RosDetection2D, Detection2DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import numpy as np
import cv2
from tf_transformations import quaternion_matrix
from PIL import Image as PILImage
from typing import Any, Optional


from stretch_floor_perception.msg import DetectedObject3D, DetectedObject3DArray
from stretch_floor_perception.srv import DetectFloorObjects3D

try:
    from tf2_ros import Buffer, TransformListener
except ImportError:
    Buffer = None
    TransformListener = None


from stretch_floor_perception.inference_backends import Detection2D, InferenceBackend

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


VLM_PROMPT = """
You are looking at a cropped image containing one main object, taken from a robot camera.

Answer these 3 questions about the main object in the crop:

1. What is this object?
2. Should a cleaning robot pick it up from the ground?
3. Should the robot grasp it from the front (along the camera/view direction) or from the top down?

Rules:
- Focus only on the main visible object in the crop.
- Be concise.
- For the object name, use a short noun phrase with no article.
  Good: "apple", "plastic bottle", "glass figure"
  Bad: "an apple", "this looks like an apple", "a small plastic bottle"
- For pickup, answer only: yes, no, or unsure
- For grasp direction, answer only: front, top, or unsure
- If the object is ambiguous, still give the best short label you can.
- Output ONLY valid JSON in this exact format:

{
  "object": "<short object name>",
  "pick_up": "yes|no|unsure",
  "grasp": "front|top|unsure"
}
""".strip()


class FloorObjectDetector(Node):
    def __init__(self):
        super().__init__("floor_object_detector")

        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("inference_backend", "yolo-darknet")
        self.declare_parameter("model_path", "")
        self.declare_parameter("config_file", "")
        self.declare_parameter("class_names_file", "")
        self.declare_parameter("use_ram_labels", False)
        self.declare_parameter("ram_model_path", "")
        self.declare_parameter("ram_image_size", 384)
        self.declare_parameter("ram_batch_size", 8)
        self.declare_parameter("publish_mask_topic", False)
        self.declare_parameter("mask_topic", "~/mask")
        self.declare_parameter("max_bbox_volume_m3", 0.0)
        self.declare_parameter("max_distance_above_floor_m", 0.0)
        self.declare_parameter("floor_z_in_base_frame_m", 0.0)
        self.declare_parameter("vlm_enabled", False)
        self.declare_parameter("vlm_base_url", "http://127.0.0.1:8080/v1")
        self.declare_parameter("vlm_api_key", "sk-no-key-required")
        self.declare_parameter("vlm_model", "")
        self.declare_parameter("vlm_prompt", VLM_PROMPT)
        self.declare_parameter("vlm_temperature", 0.0)
        self.declare_parameter("vlm_request_timeout_s", 30.0)

        self.image_with_boxes_pub = self.create_publisher(Image, "image_with_boxes", 10)
        self.centroid_pub = self.create_publisher(MarkerArray, "detected_centroids", 10)
        self.bbox3d_pub = self.create_publisher(MarkerArray, "detected_bboxes_3d", 10)
        self._prev_centroid_marker_count = 0
        self._prev_bbox3d_marker_count = 0

        self.img_bridge = CvBridge()

        self._tf_buffer = Buffer(cache_time=Duration(seconds=30.0)) if Buffer is not None else None
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True) if self._tf_buffer else None

        self.inference_backend = InferenceBackend(
            self.get_parameter("inference_backend").value,
            cfg_file=self.get_parameter("config_file").value,
            names_file=self.get_parameter("class_names_file").value,
            model=self.get_parameter("model_path").value,
            use_ram_labels=self.get_parameter("use_ram_labels").value,
            ram_model_path=self.get_parameter("ram_model_path").value,
            ram_image_size=self.get_parameter("ram_image_size").value,
            ram_batch_size=self.get_parameter("ram_batch_size").value,
            logger=self.get_logger(),
            verbose=True, # TODO make param
        )
        self.service = self.create_service(
            DetectFloorObjects3D, "detect_floor_objects_3d", self._handle_detect_service
        )
        self._vlm_enabled = bool(self.get_parameter("vlm_enabled").value)
        self._vlm_prompt = str(self.get_parameter("vlm_prompt").value).strip() or VLM_PROMPT
        self._vlm_model = str(self.get_parameter("vlm_model").value).strip()
        self._vlm_temperature = float(self.get_parameter("vlm_temperature").value)
        self._vlm_timeout_s = float(self.get_parameter("vlm_request_timeout_s").value)
        self._vlm_client = self._build_vlm_client() if self._vlm_enabled else None

    def _build_vlm_client(self):
        if not self._vlm_enabled:
            return None
        if OpenAI is None:
            raise RuntimeError("VLM support requires the 'openai' Python package.")
        if not self._vlm_model:
            raise ValueError("Parameter 'vlm_model' must be set when 'vlm_enabled' is true.")

        base_url = str(self.get_parameter("vlm_base_url").value).strip()
        api_key = str(self.get_parameter("vlm_api_key").value).strip() or "sk-no-key-required"
        self.get_logger().info(
            f"VLM enabled. Using model '{self._vlm_model}' via {base_url}"
        )
        return OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=self._vlm_timeout_s,
        )

    def _to_detection2d_array_msg(
        self,
        detections: Detection2D,
        header: Header,
        vlm_annotations: Optional[list[Optional[dict[str, str]]]] = None,
    ) -> tuple[Detection2DArray, list[dict[str, Any]]]:
        det2d_msg = Detection2DArray()
        det2d_msg.header = header
        metadata = []

        boxes_xyxy = detections.boxes_xyxy if detections.boxes_xyxy is not None else []
        ids = detections.ids if detections.ids is not None else []
        id2label = detections.id2label if detections.id2label is not None else {}
        scores = detections.scores if detections.scores is not None else []

        num_detections = max(len(boxes_xyxy), len(ids), len(scores))
        for idx in range(num_detections):
            if idx >= len(boxes_xyxy):
                continue
            box = boxes_xyxy[idx]
            class_id = ids[idx] if idx < len(ids) else idx
            score = scores[idx] if idx < len(scores) else 0.0
            box_np = np.asarray(box, dtype=np.float32).reshape(-1)
            if box_np.size < 4:
                continue

            xmin, ymin, xmax, ymax = box_np[:4]
            det = RosDetection2D()
            det.bbox.center.position.x = float((xmin + xmax) * 0.5)
            det.bbox.center.position.y = float((ymin + ymax) * 0.5)
            det.bbox.center.theta = 0.0
            det.bbox.size_x = float(xmax - xmin)
            det.bbox.size_y = float(ymax - ymin)

            result = ObjectHypothesisWithPose()
            detector_class_name = str(id2label.get(class_id, class_id))
            result.hypothesis.class_id = detector_class_name
            result.hypothesis.score = float(score)
            det.results = [result]
            det2d_msg.detections.append(det)
            metadata.append(
                {
                    "detector_class_name": detector_class_name,
                    "vlm": None if vlm_annotations is None or idx >= len(vlm_annotations) else vlm_annotations[idx],
                }
            )

        return det2d_msg, metadata

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

    def _make_detectedobject3d_msg(
        self,
        header,
        class_id,
        detector_class_name,
        pick_up,
        grasp_direction,
        score,
        centroid,
        size,
    ) -> DetectedObject3D:
        det3d = DetectedObject3D()
        det3d.header = header
        det3d.class_name = class_id
        det3d.detector_class_name = detector_class_name
        det3d.pick_up = pick_up
        det3d.grasp_direction = grasp_direction
        det3d.score = score
        det3d.centroid = Point(x=float(centroid[0]), y=float(centroid[1]), z=float(centroid[2]))
        bbox = BoundingBox3D()
        bbox.center.position = det3d.centroid
        bbox.center.orientation = self._identity_quaternion()
        bbox.size = Vector3(x=float(size[0]), y=float(size[1]), z=float(size[2]))
        det3d.bbox3d = bbox
        return det3d

    def _bbox_volume(self, size: np.ndarray) -> float:
        size_xyz = np.asarray(size, dtype=np.float32).reshape(-1)
        if size_xyz.size < 3:
            return 0.0
        return float(np.prod(np.maximum(size_xyz[:3], 0.0)))

    def _bbox_bottom_z(self, centroid: np.ndarray, size: np.ndarray) -> float:
        centroid_xyz = np.asarray(centroid, dtype=np.float32).reshape(-1)
        size_xyz = np.asarray(size, dtype=np.float32).reshape(-1)
        if centroid_xyz.size < 3 or size_xyz.size < 3:
            return float("inf")
        return float(centroid_xyz[2] - 0.5 * max(float(size_xyz[2]), 0.0))

    def _crop_bbox_image(self, image: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        if image.size == 0:
            return None
        img_h, img_w = image.shape[:2]
        box_np = np.asarray(box, dtype=np.float32).reshape(-1)
        if box_np.size < 4:
            return None

        x1, y1, x2, y2 = box_np[:4]
        x1 = int(np.clip(np.floor(x1), 0, img_w - 1))
        y1 = int(np.clip(np.floor(y1), 0, img_h - 1))
        x2 = int(np.clip(np.ceil(x2), 0, img_w))
        y2 = int(np.clip(np.ceil(y2), 0, img_h))
        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]
        return None if crop.size == 0 else crop

    def _encode_image_as_data_url(
        self,
        image_rgb: np.ndarray,
        mime_type: str = "image/jpeg",
    ) -> str:
        encode_ext = ".jpg" if mime_type == "image/jpeg" else ".png"
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        success, encoded = cv2.imencode(encode_ext, image_bgr)
        if not success:
            raise RuntimeError("Failed to encode crop for VLM request.")
        b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:{mime_type};base64,{b64}"

    def _sanitize_vlm_response(self, text: str) -> dict[str, str]:
        candidate = text.strip()
        if candidate.startswith("```"):
            lines = [line for line in candidate.splitlines() if not line.strip().startswith("```")]
            candidate = "\n".join(lines).strip()

        data = json.loads(candidate)
        obj = str(data.get("object", "")).strip() or "unknown"
        pick_up = str(data.get("pick_up", "unsure")).strip().lower()
        grasp = str(data.get("grasp", "unsure")).strip().lower()

        if pick_up not in {"yes", "no", "unsure"}:
            pick_up = "unsure"
        if grasp not in {"front", "top", "unsure"}:
            grasp = "unsure"

        return {
            "object": obj,
            "pick_up": pick_up,
            "grasp": grasp,
        }

    def _ask_vlm_about_crop(self, crop_rgb: np.ndarray) -> dict[str, str]:
        if self._vlm_client is None:
            raise RuntimeError("VLM client is not configured.")

        image_data_url = self._encode_image_as_data_url(crop_rgb)
        response = self._vlm_client.chat.completions.create(
            model=self._vlm_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._vlm_prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            temperature=self._vlm_temperature,
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content
        self.get_logger().info(text)  # DEBUG
        if isinstance(text, list):
            text = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in text
            )
        return self._sanitize_vlm_response(str(text))

    def _annotate_detections_with_vlm(
        self,
        detections: Detection2D,
    ) -> list[Optional[dict[str, str]]]:
        boxes_xyxy = detections.boxes_xyxy if detections.boxes_xyxy is not None else []
        annotations: list[Optional[dict[str, str]]] = [None] * len(boxes_xyxy)
        if self._vlm_client is None:
            return annotations

        for idx, box in enumerate(boxes_xyxy):
            crop = self._crop_bbox_image(detections.source_img, box)
            if crop is None:
                self.get_logger().warning(f"Skipping VLM call for detection {idx}: empty crop.")
                continue
            try:
                annotations[idx] = self._ask_vlm_about_crop(crop)
            except Exception as exc:
                self.get_logger().warning(
                    f"VLM request failed for detection {idx}: {exc}"
                )
        return annotations

    def _passes_3d_filters(
        self,
        class_id: str,
        centroid: np.ndarray,
        size: np.ndarray,
        max_bbox_volume_m3: float,
        max_distance_above_floor_m: float,
        floor_z_in_base_frame_m: float,
    ) -> bool:
        bbox_volume_m3 = self._bbox_volume(size)
        if max_bbox_volume_m3 > 0.0 and bbox_volume_m3 > max_bbox_volume_m3:
            self.get_logger().info(
                f"Filtering detection '{class_id}': bbox volume "
                f"{bbox_volume_m3:.4f} m^3 exceeds limit {max_bbox_volume_m3:.4f} m^3."
            )
            return False

        if max_distance_above_floor_m > 0.0:
            bbox_bottom_z = self._bbox_bottom_z(centroid, size)
            distance_above_floor_m = bbox_bottom_z - floor_z_in_base_frame_m
            if distance_above_floor_m > max_distance_above_floor_m:
                self.get_logger().info(
                    f"Filtering detection '{class_id}': bbox bottom is "
                    f"{distance_above_floor_m:.4f} m above floor, exceeding limit "
                    f"{max_distance_above_floor_m:.4f} m."
                )
                return False

        return True

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
        return marker

    def _empty_det3d_msg(self, stamp=None, frame_id=None) -> DetectedObject3DArray:
        det3d_msg = DetectedObject3DArray()
        det3d_msg.header = Header()
        if stamp is not None:
            det3d_msg.header.stamp = Time() #stamp   # TODO Fix. Setting to zero because timestamp is too old for rviz
        if frame_id is not None:
            det3d_msg.header.frame_id = frame_id
        det3d_msg.detections = []
        return det3d_msg

    def _handle_detect_service(self, request, response):
        det3d_msg, det2d_msg, image_with_boxes, centroid_markers, bbox3d_markers = self._process_images(
            request.color_msg, request.depth_msg, request.info_msg
        )
        response.detections = det3d_msg

        if image_with_boxes is not None:
            if isinstance(image_with_boxes, PILImage.Image):
                image_with_boxes = np.array(image_with_boxes)
            image_with_boxes_msg = self.img_bridge.cv2_to_imgmsg(
                image_with_boxes, encoding="rgb8"
            )
            image_with_boxes_msg.header = det2d_msg.header
            self.image_with_boxes_pub.publish(image_with_boxes_msg)
        if centroid_markers is not None:
            self.centroid_pub.publish(centroid_markers)
        if bbox3d_markers is not None:
            self.bbox3d_pub.publish(bbox3d_markers)
        return response

    def _decode_depth_compressed_image(self, depth_msg: CompressedImage):
        # Try cv_bridge first; fall back to manual handling for compressedDepth payloads.
        try:
            depth_img = self.img_bridge.compressed_imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            if depth_img is None:
                depth_fmt, compr_type = depth_msg.format.split(';')
                if depth_fmt.strip() == "16UC1" and compr_type.strip() == "compressedDepth":
                    self.get_logger().warning(f"Using custom depth decoding for {depth_fmt}")
                    PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
                    raw = bytes(depth_msg.data)
                    png_start = raw.find(PNG_MAGIC)
                    if png_start == -1:
                        png_start = 0
                    raw_png = raw[png_start:]
                
                    depth_bytes = np.frombuffer(raw_png, dtype=np.uint8)
                    depth_img = cv2.imdecode(depth_bytes, cv2.IMREAD_UNCHANGED)
                    if depth_img is not None:
                        return depth_img
                    else:
                        raise RuntimeError(f"Could not decode depth compressed image with format '{depth_msg.format}'")
                else:
                    raise RuntimeError(f"Could not decode depth compressed image with format '{depth_msg.format}'")
            else:
                return depth_img

        except Exception as e:
            raise RuntimeError(f"Could not decode depth compressed image with format '{depth_msg.format}'")


    def _process_images(
        self, color_msg: CompressedImage, depth_msg: CompressedImage, info_msg: CameraInfo
    ):
        t_start= time()
        self.get_logger().info("Image received. Starting inference.")
        color_img = self.img_bridge.compressed_imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
        color_img = cv2.rotate(color_img, cv2.ROTATE_90_CLOCKWISE)
        color_img, pad_left, pad_top = self._pad_to_center_square(color_img)

        detections = self.inference_backend.infer(color_img, threshold=0.85)
        self.get_logger().info("Got inference result")
        vlm_annotations = self._annotate_detections_with_vlm(detections)
        det2d_msg, det2d_metadata = self._to_detection2d_array_msg(
            detections,
            color_msg.header,
            vlm_annotations=vlm_annotations,
        )
        image_with_boxes = detections.result_image()
        det2d_msg.header = color_msg.header


        det3d_msg = DetectedObject3DArray()
        det3d_msg.header = Header()
        det3d_msg.header.stamp = color_msg.header.stamp
        det3d_msg.header.frame_id = self.get_parameter("base_frame").value

        depth_img = self._decode_depth_compressed_image(depth_msg)
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
                base_frame, camera_frame, rclpy.time.Time(), timeout=Duration(seconds=1.0)
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
        max_bbox_volume_m3 = float(self.get_parameter("max_bbox_volume_m3").value)
        max_distance_above_floor_m = float(
            self.get_parameter("max_distance_above_floor_m").value
        )
        floor_z_in_base_frame_m = float(
            self.get_parameter("floor_z_in_base_frame_m").value
        )

        n=0
        for det_idx, det in enumerate(det2d_msg.detections):
            if not det.results:
                continue
            best = max(det.results, key=lambda r: r.hypothesis.score)
            detector_class_name = best.hypothesis.class_id
            vlm_annotation = (
                det2d_metadata[det_idx].get("vlm")
                if det_idx < len(det2d_metadata)
                else None
            )
            class_id = (
                vlm_annotation.get("object", detector_class_name)
                if vlm_annotation is not None
                else detector_class_name
            )
            pick_up = (
                vlm_annotation.get("pick_up", "unsure")
                if vlm_annotation is not None
                else "unsure"
            )
            grasp_direction = (
                vlm_annotation.get("grasp", "unsure")
                if vlm_annotation is not None
                else "unsure"
            )
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
            if not self._passes_3d_filters(
                class_id=class_id,
                centroid=centroid,
                size=size,
                max_bbox_volume_m3=max_bbox_volume_m3,
                max_distance_above_floor_m=max_distance_above_floor_m,
                floor_z_in_base_frame_m=floor_z_in_base_frame_m,
            ):
                continue

            det3d = self._make_detectedobject3d_msg(
                det3d_msg.header,
                class_id,
                detector_class_name,
                pick_up,
                grasp_direction,
                score,
                centroid,
                size,
            )
            det3d_msg.detections.append(det3d)
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
            )
            bbox3d_markers.markers.append(bbox_marker)
            n += 1
        for marker_id in range(n, self._prev_centroid_marker_count):
            stale_marker = Marker()
            stale_marker.header = det3d_msg.header
            stale_marker.ns = "detected_centroids"
            stale_marker.id = marker_id
            stale_marker.action = Marker.DELETE
            centroid_markers.markers.append(stale_marker)

        for marker_id in range(n, self._prev_bbox3d_marker_count):
            stale_marker = Marker()
            stale_marker.header = det3d_msg.header
            stale_marker.ns = "detected_bboxes_3d"
            stale_marker.id = marker_id
            stale_marker.action = Marker.DELETE
            bbox3d_markers.markers.append(stale_marker)

        self._prev_centroid_marker_count = n
        self._prev_bbox3d_marker_count = n
        self.get_logger().info(f"DONE")
        self.get_logger().warning(f"Took: {time()-t_start} seconds.")
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
