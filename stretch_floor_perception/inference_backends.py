import numpy as np
import cv2
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose


class InferenceBackend:
    """Build and dispatch a named inference backend."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self._kwargs = kwargs
        self.backend = self._build_backend()

    def _build_backend(self):
        builders = {
            "yolo-darknet": DarkNetBackend,
            "darknet": DarkNetBackend,
            "onnxruntime": OnnxRuntimeBackend,
        }
        try:
            builder = builders[self.model_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported inference model: {self.model_name}") from exc
        return builder(**self._kwargs)

    def infer(self, image: np.array):
        return self.backend.infer(image)


class OnnxRuntimeBackend:
    """Placeholder for ONNX Runtime inference."""

    def __init__(self, model_path: str, logger=None, **_kwargs):
        self.model_path = model_path
        self._logger = logger

    def infer(self, image: np.array):
        # TODO: return detections and optional masks using onnxruntime.
        return []


class DarkNetBackend:
    """DarkNet inference backend"""

    def __init__(self, 
                 cfg_file: str, 
                 names_file: str, 
                 weights_file: str, 
                 logger=None,
                 verbose=False, 
                 **_kwargs):
        try:
            #import darknet  # type: ignore
            from stretch_floor_perception import darknet
        except ImportError as e:
            raise RuntimeError(
                "DarkNetBackend requires the 'darknet' Python bindings"
            ) from e

        self.darknet = darknet
        self._logger = logger
        self.verbose = verbose
        if self.verbose:
            darknet.set_verbose(True)
        # load_net_custom has 4 params (3rd one not documented)
        # (cfg_file, weights_file, clear, batch)
        self.network = darknet.load_net_custom(cfg_file.encode("ascii"), 
                                               weights_file.encode("ascii"),0,1)
        self.class_names = open(names_file).read().splitlines()
        self.colours = darknet.class_colors(self.class_names)
        self.prediction_threshold = 0.5
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
    
    def close(self):
        if not self._closed:
            self.darknet.free_network_ptr(self.network)
            self.network = None
            self._closed = True

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def infer(self, image: np.array):
        image_resized = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        darknet_image = self.darknet.make_image(self.width, self.height, 3)
        self.darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = self.darknet.detect_image(self.network, 
                                               self.class_names, 
                                               darknet_image, 
                                               thresh=self.prediction_threshold)
        self.darknet.free_image(darknet_image)

        if self.verbose:
            self.darknet.print_detections(detections, True)
        
        image_with_boxes = self.darknet.draw_boxes(detections, image_resized, self.colours)

        scale_x = image.shape[1] / self.width
        scale_y = image.shape[0] / self.height
        det2d_msg = Detection2DArray()
        for label, confidence, (cx, cy, w, h) in detections:
            det = Detection2D()
            det.bbox.center.position.x = cx * scale_x
            det.bbox.center.position.y = cy * scale_y
            det.bbox.center.theta = 0.0
            det.bbox.size_x = w * scale_x
            det.bbox.size_y = h * scale_y
            result = ObjectHypothesisWithPose()
            result.hypothesis.class_id = label
            result.hypothesis.score = float(confidence)
            det.results = [result]
            det2d_msg.detections.append(det)
        return det2d_msg, image_with_boxes
