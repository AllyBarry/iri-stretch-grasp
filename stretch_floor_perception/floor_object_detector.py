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
from stretch_floor_perception.srv import DetectFloorObjects3D, DetectObjects2D

try:
    from tf2_ros import Buffer, TransformListener
except ImportError:
    Buffer = None
    TransformListener = None


from stretch_floor_perception.inference_backends import Detection2D, InferenceBackend


class UltralyticsYolo:
    """Fine-tuned / any ultralytics YOLO model (.pt, .onnx, openvino dir). Returns the same
    Detection2D the other backends do, with the model's class names as labels."""

    def __init__(self, model_path, conf, logger):
        import os
        from ultralytics import YOLO
        self.model = YOLO(os.path.expanduser(model_path))
        self.conf = conf
        self.logger = logger
        logger.info(f"YOLO model '{model_path}' loaded, classes {list(self.model.names.values())}, conf >= {conf}")

    def infer(self, image, threshold=None):  # node's threshold (0.85, for FastSAM) is ignored
        bgr = np.ascontiguousarray(image[..., ::-1])  # node passes RGB, ultralytics expects BGR
        # retina_masks: masks at full image resolution. They are used as rasters, not polygons: an
        # object split in two by something in front (basket) would otherwise be filled across the gap.
        pred = self.model(bgr, conf=self.conf, verbose=False, retina_masks=True)[0]
        masks = None
        if getattr(pred, "masks", None) is not None and pred.masks.data is not None:
            masks = pred.masks.data.cpu().numpy() > 0.5
        det2d = Detection2D(source_img=image, boxes_xyxy=[], masks=[] if masks is not None else None)
        if pred.boxes is None or len(pred.boxes) == 0:
            return det2d
        xyxy = pred.boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = pred.boxes.conf.cpu().numpy()
        classes = pred.boxes.cls.cpu().numpy().astype(int)
        for j, (box, c, k) in enumerate(zip(xyxy, confs, classes)):
            label = str(pred.names[int(k)])
            self.logger.info(f"YOLO: {label} {float(c):.2f}")
            m = None
            if masks is not None and j < len(masks):
                m = masks[j]
                if m.shape != image.shape[:2]:
                    m = cv2.resize(m.astype(np.uint8), (image.shape[1], image.shape[0]),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
            det2d.add_detection(id=label, score=float(c), box=box, mask=m)
        return det2d

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
- Tall vertical objects should be grasp from the front (e.g. bottle, cup)
- Short elongated objects shold be grasp from the top (e.g. pen, coffee box)
- If the object is ambiguous, still give the best short label you can.
- Output ONLY valid JSON in this exact format:

{
  "object": "<short object name>",
  "pick_up": "yes|no|unsure",
  "grasp": "front|top|unsure"
}
""".strip()


def _iou(a, b):
    ax1, ay1, ax2, ay2 = [float(v) for v in a[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in b[:4]]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class YoloFastSamHybrid:
    """Names from the fine-tuned YOLO, geometry from FastSAM: each YOLO detection takes the box
    of the FastSAM segment that overlaps it most (IoU >= min_iou), else keeps its own box."""

    def __init__(self, yolo, fastsam, min_iou, logger):
        self.yolo, self.fastsam, self.min_iou, self.logger = yolo, fastsam, min_iou, logger

    def infer(self, image, threshold=0.85):
        y = self.yolo.infer(image)
        f = self.fastsam.infer(image, threshold=threshold)
        fboxes = [b for b in (f.boxes_xyxy or []) if b is not None]
        out = Detection2D(source_img=image, boxes_xyxy=[], masks=None)
        used = set()
        for label, score, ybox in zip(y.ids, y.scores, y.boxes_xyxy or []):
            best_j, best = None, 0.0
            for j, fb in enumerate(fboxes):
                if j in used:
                    continue
                v = _iou(ybox, fb)
                if v > best:
                    best_j, best = j, v
            if best_j is not None and best >= self.min_iou:
                used.add(best_j)
                box = np.asarray(fboxes[best_j], dtype=np.float32)
                self.logger.info(f"hybrid: {label} -> FastSAM segment (IoU {best:.2f})")
            else:
                box = np.asarray(ybox, dtype=np.float32)
                self.logger.info(f"hybrid: {label} -> no FastSAM match (best IoU {best:.2f}), using YOLO box")
            out.add_detection(id=label, score=score, box=box, mask=None)
        return out


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
        self.declare_parameter("max_distance_above_floor_m", 0.3)
        self.declare_parameter("floor_z_in_base_frame_m", 0.0)
        self.declare_parameter("vlm_enabled", False)
        self.declare_parameter("vlm_base_url", "http://127.0.0.1:8080/v1")
        self.declare_parameter("vlm_api_key", "sk-no-key-required")
        self.declare_parameter("vlm_model", "")
        self.declare_parameter("vlm_prompt", VLM_PROMPT)
        self.declare_parameter("vlm_temperature", 0.0)
        self.declare_parameter("vlm_request_timeout_s", 30.0)
        self.declare_parameter("yolo_conf", 0.35)
        self.declare_parameter("yolo_model_path", "")   # YOLO weights for inference_backend:=yolo-fastsam
        self.declare_parameter("hybrid_min_iou", 0.3)   # min YOLO/FastSAM box overlap to take FastSAM's box
        # class:grasp pairs; listed classes get pick_up=yes and that grasp when no VLM is used
        self.declare_parameter("class_grasp", "garlic:top,lemon:top,onion:top,tomato:top,"
                                              "salt:front,mustard:front,balsamic:front,"
                                              "olive oil:front,chick peas:front")
        # Picking the object's own points out of the 2D box (the box also covers surface and background)
        self.declare_parameter("box_inner_fraction", 0.6)   # central part of the box used for the object depth
        self.declare_parameter("depth_band_m", 0.08)        # keep box points within this of that depth
        self.declare_parameter("surface_clearance_m", 0.01) # drop points this close to the supporting surface
        self.declare_parameter("min_object_points", 20)
        # Fraction of the object's radius to push the centroid away from the camera (it only sees the
        # near side). 0 disables. Read on every scan, so `ros2 param set` takes effect immediately.
        self.declare_parameter("near_side_correction", 0.0)

        self.image_with_boxes_pub = self.create_publisher(Image, "image_with_boxes", 10)
        self.centroid_pub = self.create_publisher(MarkerArray, "detected_centroids", 10)
        self.detections_pub = self.create_publisher(DetectedObject3DArray, "detected_objects", 10)
        self.bbox3d_pub = self.create_publisher(MarkerArray, "detected_bboxes_3d", 10)
        self._prev_centroid_marker_count = 0
        self._prev_bbox3d_marker_count = 0

        self.img_bridge = CvBridge()

        self._tf_buffer = Buffer(cache_time=Duration(seconds=30.0)) if Buffer is not None else None
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True) if self._tf_buffer else None

        if self.get_parameter("inference_backend").value == "yolo-fastsam":
            self.inference_backend = YoloFastSamHybrid(
                UltralyticsYolo(
                    self.get_parameter("yolo_model_path").value,
                    float(self.get_parameter("yolo_conf").value),
                    self.get_logger(),
                ),
                InferenceBackend(
                    "fastsam",
                    cfg_file=self.get_parameter("config_file").value,
                    names_file=self.get_parameter("class_names_file").value,
                    model=self.get_parameter("model_path").value,
                    use_ram_labels=False,
                    ram_model_path=self.get_parameter("ram_model_path").value,
                    ram_image_size=self.get_parameter("ram_image_size").value,
                    ram_batch_size=self.get_parameter("ram_batch_size").value,
                    logger=self.get_logger(),
                    verbose=True,
                ),
                float(self.get_parameter("hybrid_min_iou").value),
                self.get_logger(),
            )
        elif self.get_parameter("inference_backend").value == "yolo-ultralytics":
            self.inference_backend = UltralyticsYolo(
                self.get_parameter("model_path").value,
                float(self.get_parameter("yolo_conf").value),
                self.get_logger(),
            )
        else:
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
        self.service_2d = self.create_service(
            DetectObjects2D, "detect_objects_2d", self._handle_detect_2d_service
        )
        self._vlm_enabled = bool(self.get_parameter("vlm_enabled").value)
        self._vlm_prompt = str(self.get_parameter("vlm_prompt").value).strip() or VLM_PROMPT
        self._vlm_model = str(self.get_parameter("vlm_model").value).strip()
        self._vlm_temperature = float(self.get_parameter("vlm_temperature").value)
        self._vlm_timeout_s = float(self.get_parameter("vlm_request_timeout_s").value)
        self._vlm_client = self._build_vlm_client() if self._vlm_enabled else None
        self._class_grasp = {}
        for item in str(self.get_parameter("class_grasp").value).split(","):
            if ":" in item:
                k, v = item.split(":", 1)
                self._class_grasp[k.strip()] = v.strip().lower()
        self._box_inner_fraction = float(self.get_parameter("box_inner_fraction").value)
        self._depth_band_m = float(self.get_parameter("depth_band_m").value)
        self._surface_clearance_m = float(self.get_parameter("surface_clearance_m").value)
        self._min_object_points = int(self.get_parameter("min_object_points").value)

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
                    "idx": idx,
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

    def _backproject_depth_crop(self, depth_im, xmin, xmax, ymin, ymax, fx, fy, cx, cy,
                                z_min=0.0, z_max=np.inf):
        depth_crop = depth_im[ymin:ymax, xmin:xmax]
        valid = np.isfinite(depth_crop) & (depth_crop > max(z_min, 0.0)) & (depth_crop <= z_max)
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

        # The centre of the box is mostly the object, so its median depth is the object's depth.
        # Keep only box points near it, which drops the background and far parts of the surface.
        ixmin, ixmax, iymin, iymax = self._clipped_window(
            cx_pix, cy_pix, box_size_x * self._box_inner_fraction, box_size_y * self._box_inner_fraction,
            width, height)
        inner = depth_im[iymin:iymax, ixmin:ixmax]
        inner = inner[np.isfinite(inner) & (inner > 0.0)]
        if inner.size == 0:
            self.get_logger().debug("Skipping detection: no valid depth values in centre of bounding box")
            return None
        z_obj = float(np.median(inner))

        points_cam = self._backproject_depth_crop(depth_im, xmin, xmax, ymin, ymax, fx, fy, cx, cy,
                                                  z_min=z_obj - self._depth_band_m,
                                                  z_max=z_obj + self._depth_band_m)
        if points_cam is None:
            self.get_logger().debug("Skipping detection: no valid depth values in bounding box")
            return None
        
        if points_cam.shape[0] > max_points:
            step = int(np.ceil(points_cam.shape[0] / max_points))
            points_cam = points_cam[::step]
    
        return points_cam


    def _estimate_object_3d(self, points_base: np.ndarray, camera_base: np.ndarray,
                            near_side_correction: float = 1.0, surface_z: Optional[float] = None):
        """Centre and size of the object from its depth points (base frame, z up).

        surface_z: height of the surface it stands on, when known (mask path). Then all points are
        the object's. Without it (box path) the points also contain surface, which is removed here.

        Returns (centroid, size, num_object_points, near_side_push_m, yaw). The box is aligned
        with the object's long axis seen from above: yaw is that axis's angle about z in the base
        frame, and size is (long, short, height). Medians and percentiles are used so stray edge
        and background points don't move the result.
        """
        if surface_z is not None:
            obj = points_base
            top_z = float(np.percentile(obj[:, 2], 95.0))
            bottom_z = surface_z
        else:
            # Whatever surface is left in the box is its lowest part, so drop points close to a low
            # percentile of the heights. Keep all points if too few would be left (no surface visible).
            surface_z = float(np.percentile(points_base[:, 2], 10.0))
            above = points_base[:, 2] > surface_z + self._surface_clearance_m
            on_surface = int(np.count_nonzero(above)) >= self._min_object_points
            obj = points_base[above] if on_surface else points_base

            top_z = float(np.percentile(obj[:, 2], 95.0))
            bottom_z = surface_z if on_surface else float(np.percentile(obj[:, 2], 5.0))

        # The camera only sees the near side of the object, so the median of its points is short
        # of the centre. Push it away from the camera by the object's radius (its width across the
        # view), scaled by how horizontal the view is (looking straight down sees no near side).
        xy = np.median(obj[:, :2], axis=0)
        view = np.append(xy, 0.5 * (top_z + bottom_z)) - camera_base
        view_xy = view[:2]
        horiz = float(np.linalg.norm(view_xy))
        width = 0.0
        push_m = 0.0
        if horiz > 1e-6:
            along = view_xy / horiz
            across = np.array([-along[1], along[0]])
            proj = obj[:, :2] @ across
            width = float(np.percentile(proj, 95.0) - np.percentile(proj, 5.0))
            push_m = near_side_correction * 0.5 * width * horiz / float(np.linalg.norm(view))
            xy = xy + along * push_m

        # Long axis of the footprint, so the box follows the object when it lies rotated.
        yaw = 0.0
        centered_xy = obj[:, :2] - np.median(obj[:, :2], axis=0)
        if centered_xy.shape[0] >= 3:
            eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered_xy, rowvar=False))
            major = eigenvectors[:, int(np.argmax(eigenvalues))]
            yaw = float(np.arctan2(major[1], major[0]))
        major_axis = np.array([np.cos(yaw), np.sin(yaw)])
        minor_axis = np.array([-major_axis[1], major_axis[0]])
        proj_major = centered_xy @ major_axis
        proj_minor = centered_xy @ minor_axis
        major_len = float(np.percentile(proj_major, 95.0) - np.percentile(proj_major, 5.0))
        minor_len = float(np.percentile(proj_minor, 95.0) - np.percentile(proj_minor, 5.0))
        size = np.array([max(major_len, minor_len), minor_len, max(top_z - bottom_z, 0.0)])
        centroid = np.array([xy[0], xy[1], 0.5 * (top_z + bottom_z)])
        return centroid, size, obj.shape[0], push_m, yaw

    def _pixels_to_points(self, xs, ys, pad_left, pad_top, depth_im, intrinsics, max_points=20000):
        """Camera-frame 3D points for pixels of the rotated, padded detector image."""
        fx, fy, cx, cy = intrinsics
        h, w = depth_im.shape[:2]
        x, y = xs - pad_left, ys - pad_top
        u, v = y, (h - 1) - x                     # undo the 90 degree clockwise rotation
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        u, v = u[ok], v[ok]
        z = depth_im[v, u].astype(np.float32)
        good = np.isfinite(z) & (z > 0)
        u, v, z = u[good].astype(np.float32), v[good].astype(np.float32), z[good]
        pts = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=1)
        if pts.shape[0] > max_points:
            pts = pts[:: int(np.ceil(pts.shape[0] / max_points))]
        return pts

    def _mask_points(self, det_idx, metadata, detections, pad_left, pad_top, depth_im, info_msg,
                     max_points=20000):
        """3D points (camera frame) from the depth pixels under a segmentation mask, so the table and
        background inside the box are never used, plus the points of a thin ring just outside the
        mask, which show the surface the object stands on. (None, None) when there is no usable mask."""
        masks = getattr(detections, "masks", None)
        if not masks or det_idx >= len(metadata):
            return None, None
        midx = metadata[det_idx].get("idx")
        if midx is None or midx >= len(masks) or masks[midx] is None:
            return None, None
        intr = self._get_intrinsics(info_msg)
        if intr is None:
            return None, None
        full = masks[midx].astype(np.uint8)
        mask = cv2.erode(full, np.ones((5, 5), np.uint8))  # stay off the edges
        ys, xs = np.nonzero(mask)
        if len(xs) < 20:
            return None, None
        pts = self._pixels_to_points(xs, ys, pad_left, pad_top, depth_im, intr, max_points)
        if pts.shape[0] < 20:
            return None, None
        # Depth along a mask edge can belong to whatever is behind, so keep the object's depth range.
        z_med = float(np.median(pts[:, 2]))
        pts = pts[np.abs(pts[:, 2] - z_med) <= self._depth_band_m]
        # Ring 3-8 px outside the mask (the gap skips the edge pixels).
        ring = (cv2.dilate(full, np.ones((17, 17), np.uint8)) > 0) & ~(cv2.dilate(full, np.ones((7, 7), np.uint8)) > 0)
        rys, rxs = np.nonzero(ring)
        ring_pts = self._pixels_to_points(rxs, rys, pad_left, pad_top, depth_im, intr, max_points)
        self.get_logger().info(f"mask: {pts.shape[0]} depth points, ring: {ring_pts.shape[0]}")
        return pts, ring_pts

    def _surface_below(self, obj_base: np.ndarray, ring_base: Optional[np.ndarray],
                       max_gap_m: float = 0.10, near_m: float = 0.10) -> float:
        """Height of the surface the object stands on, from the ring of points around its mask.

        Only ring points within near_m of the object (horizontally) count, so the floor or a wall
        seen behind the object is ignored. Neighbours in a basket are higher than what the object
        rests on, so a low percentile is used. Falls back to the object's own lowest points when the
        ring shows nothing usable (e.g. the bottom is hidden).
        """
        obj_low = float(np.percentile(obj_base[:, 2], 5.0))
        if ring_base is None or ring_base.shape[0] < self._min_object_points:
            return obj_low
        centre = np.median(obj_base[:, :2], axis=0)
        reach = near_m + 0.5 * float(np.max(np.ptp(obj_base[:, :2], axis=0)))
        near = ring_base[np.linalg.norm(ring_base[:, :2] - centre, axis=1) <= reach]
        if near.shape[0] < self._min_object_points:
            return obj_low
        surface_z = float(np.percentile(near[:, 2], 20.0))
        if surface_z > obj_low or obj_low - surface_z > max_gap_m:
            return obj_low
        return surface_z

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
        yaw=0.0,
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
        bbox.center.orientation = Quaternion(
            x=0.0, y=0.0, z=float(np.sin(0.5 * yaw)), w=float(np.cos(0.5 * yaw)))
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
            self.detections_pub.publish(det3d_msg)
            self.centroid_pub.publish(centroid_markers)
        if bbox3d_markers is not None:
            self.bbox3d_pub.publish(bbox3d_markers)
        return response

    def _handle_detect_2d_service(self, request, response):
        # 2D detections only, e.g. for the grasp nodes to check what is in the gripper,
        # so no depth, VLM or floor filtering.
        color_msg = request.color_msg
        color_img = self.img_bridge.compressed_imgmsg_to_cv2(color_msg, desired_encoding="rgb8")
        height = color_img.shape[0]
        rotated = cv2.rotate(color_img, cv2.ROTATE_90_CLOCKWISE)
        padded, pad_left, pad_top = self._pad_to_center_square(rotated)
        detections = self.inference_backend.infer(padded, threshold=0.85)
        det2d_msg, _ = self._to_detection2d_array_msg(detections, color_msg.header)

        # Map the boxes from the rotated, padded image back to the image as received.
        # ROTATE_90_CLOCKWISE maps box coordinates (x, y) to (height - y, x).
        for det in det2d_msg.detections:
            x_rot = det.bbox.center.position.x - pad_left
            y_rot = det.bbox.center.position.y - pad_top
            det.bbox.center.position.x = float(y_rot)
            det.bbox.center.position.y = float(height - x_rot)
            det.bbox.size_x, det.bbox.size_y = det.bbox.size_y, det.bbox.size_x

        response.detections = det2d_msg
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
            if vlm_annotation is None and detector_class_name in self._class_grasp:
                pick_up = "yes"
                grasp_direction = self._class_grasp[detector_class_name]

            cx_pix = det.bbox.center.position.x - pad_left
            cy_pix = det.bbox.center.position.y - pad_top

            pts3d_window_cam = self._get_3dpts_in_window(cx_pix, 
                                                         cy_pix, 
                                                         det.bbox.size_x, 
                                                         det.bbox.size_y, 
                                                         depth_m, 
                                                         info_msg,
                                                         max_points = 20000)
            mask_pts, ring_pts = self._mask_points(det_idx, det2d_metadata, detections, pad_left, pad_top,
                                                   depth_m, info_msg)
            if mask_pts is not None:
                pts3d_window_cam = mask_pts
            if pts3d_window_cam is None:
                continue

            points_base = pts3d_window_cam @ rot.T[:3, :3] + trans[:3]
            surface_z = None
            if mask_pts is not None:
                ring_base = ring_pts @ rot.T[:3, :3] + trans[:3] if ring_pts is not None and len(ring_pts) else None
                surface_z = self._surface_below(points_base, ring_base)
            centroid, size, num_points, push_m, yaw = self._estimate_object_3d(
                points_base, trans[:3], float(self.get_parameter("near_side_correction").value),
                surface_z=surface_z,
            )
            if num_points < self._min_object_points:
                self.get_logger().info(
                    f"Skipping detection '{class_id}': only {num_points} depth points on the object."
                )
                continue
            self.get_logger().info(
                f"'{class_id}': centroid ({centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}) "
                f"size ({size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}), yaw {np.degrees(yaw):.0f} deg, "
                f"pushed {push_m * 100:.1f} cm away from camera"
            )
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
                yaw,
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
            label_marker = self._make_marker(
                det3d_msg.header,
                "detected_labels",
                n,
                Marker.TEXT_VIEW_FACING,
                Point(
                    x=det3d.centroid.x,
                    y=det3d.centroid.y,
                    z=det3d.centroid.z + det3d.bbox3d.size.z / 2.0 + 0.06,
                ),
                self._identity_quaternion(),
                Vector3(x=0.0, y=0.0, z=0.06),
                (1.0, 1.0, 1.0),
                1.0,
            )
            label_marker.text = f"{class_id} [{pick_up}/{grasp_direction}]"
            centroid_markers.markers.append(label_marker)
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
            label_stale = Marker()
            label_stale.header = det3d_msg.header
            label_stale.ns = "detected_labels"
            label_stale.id = marker_id
            label_stale.action = Marker.DELETE
            centroid_markers.markers.append(label_stale)

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
