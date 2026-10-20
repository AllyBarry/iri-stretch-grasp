from abc import ABC, abstractmethod
from typing import Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image


class Detection2D:
    """Container for 2D detections.

    boxes_xyxy stores bounding boxes in image pixel coordinates as
    [x_min, y_min, x_max, y_max].
    """

    def __init__(
        self,
        source_img: np.ndarray,
        ids: Optional[list] = None,
        id2label: Optional[dict] = None,
        scores: Optional[list] = None,
        boxes_xyxy: Optional[list] = None,
        masks: Optional[list] = None,
    ):
        self.source_img = source_img
        self.ids = [] if ids is None else ids
        self.id2label = {} if id2label is None else dict(id2label)
        self.scores = [] if scores is None else scores
        self.boxes_xyxy = boxes_xyxy
        self.masks = masks

    def add_detection(
        self,
        id: Any,
        score: float,
        box: Optional[np.ndarray],
        mask: Optional[np.ndarray],
        label: Optional[str] = None,
    ):
        self.ids.append(id)
        if label is not None:
            self.id2label[id] = label
        self.scores.append(score)
        if box is not None and self.boxes_xyxy is not None:
            self.boxes_xyxy.append(box)
        if mask is not None and self.masks is not None:
            self.masks.append(mask)

    def result_image(self, threshold: Optional[float] = None):
        image = self.source_img.copy()
        if image.size == 0:
            return image

        scores = self.scores if self.scores is not None else []
        ids = self.ids if self.ids is not None else []
        boxes_xyxy = self.boxes_xyxy if self.boxes_xyxy is not None else []
        masks = self.masks if self.masks is not None else []
        num_detections = max(len(ids), len(scores), len(boxes_xyxy), len(masks))

        img_h, img_w = image.shape[:2]
        alpha_num = 77
        alpha_den = 255

        for i in range(num_detections):
            score = float(scores[i]) if i < len(scores) else None
            if threshold is not None and score is not None and score < threshold:
                continue

            label_id = ids[i] if i < len(ids) else i
            label_name = self.id2label.get(label_id, label_id)
            color_seed = sum(ord(ch) for ch in str(label_name))
            color = (
                (37 * (color_seed + 3)) % 256,
                (17 * (color_seed + 7)) % 256,
                (29 * (color_seed + 11)) % 256,
            )

            if i < len(masks) and masks[i] is not None:
                mask = np.asarray(masks[i])
                if mask.ndim == 3:
                    mask = np.squeeze(mask)
                if mask.shape[:2] == (img_h, img_w):
                    if mask.dtype != np.bool_:
                        mask = mask > (threshold if threshold is not None else 0.5)
                    if np.any(mask):
                        masked = image[mask]
                        if masked.size:
                            image[mask] = (
                                (masked.astype(np.uint16) * (alpha_den - alpha_num)
                                 + np.asarray(color, dtype=np.uint16) * alpha_num)
                                // alpha_den
                            ).astype(np.uint8)

            if i < len(boxes_xyxy) and boxes_xyxy[i] is not None:
                box = np.asarray(boxes_xyxy[i], dtype=np.float32).reshape(-1)
                if box.size >= 4:
                    x1, y1, x2, y2 = box[:4]
                    x1 = int(np.clip(np.floor(x1), 0, img_w - 1))
                    y1 = int(np.clip(np.floor(y1), 0, img_h - 1))
                    x2 = int(np.clip(np.ceil(x2), 0, img_w - 1))
                    y2 = int(np.clip(np.ceil(y2), 0, img_h - 1))
                    if x2 > x1 and y2 > y1:
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        label_text = (
                            str(label_name) if score is None else f"{label_name}: {score:.2f}"
                        )
                        (text_w, text_h), baseline = cv2.getTextSize(
                            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                        )
                        text_y = text_h + 4 if y1 <= text_h + 4 else y1
                        top = max(text_y - text_h - baseline - 4, 0)
                        cv2.rectangle(
                            image,
                            (x1, top),
                            (min(x1 + text_w + 6, img_w - 1), text_y + baseline),
                            color,
                            thickness=-1,
                        )
                        cv2.putText(
                            image,
                            label_text,
                            (x1 + 3, text_y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

        return image


class InferenceBackend:
    """Build and dispatch a named inference backend."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self._kwargs = kwargs
        self.backend = self._build_backend()

    def _build_backend(self):
        builders = {
            "yolo-darknet": DarkNetBackend,
            "detr": DetrBackend,
            "onnxruntime": OnnxRuntimeBackend,
            "fastsam": FastSamBackend,
            "ultralytics-fastsam": FastSamBackend,
        }
        try:
            builder = builders[self.model_name]
        except KeyError as exc:
            raise ValueError(f"Unsupported inference model: {self.model_name}") from exc
        return builder(**self._kwargs)

    def infer(self, image: np.ndarray, threshold: float):
        return self.backend.infer(image, threshold)


class BaseInferenceBackend(ABC):
    @abstractmethod
    def infer(self, image: np.ndarray, threshold: float):
        pass


class OnnxRuntimeBackend(BaseInferenceBackend):
    """Placeholder for ONNX Runtime inference."""

    def __init__(self, model_path: str, logger=None, **_kwargs):
        self.model_path = model_path
        self._logger = logger

    def infer(self, image: np.ndarray, threshold: float):
        return Detection2D(source_img=image, boxes_xyxy=[], masks=None)


class DarkNetBackend(BaseInferenceBackend):
    """DarkNet inference backend."""

    def __init__(
        self,
        cfg_file: str,
        names_file: str,
        model: str,
        logger=None,
        verbose=False,
        **_kwargs,
    ):
        try:
            from stretch_floor_perception import darknet
        except ImportError as e:
            raise RuntimeError(
                "DarkNetBackend requires the 'darknet' Python bindings"
            ) from e

        self.darknet = darknet
        self._logger = logger
        self.verbose = verbose
        self._closed = False
        if self.verbose:
            darknet.set_verbose(True)
        self.network = darknet.load_net_custom(
            cfg_file.encode("ascii"),
            model.encode("ascii"),
            0,
            1,
        )
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

    def infer(self, image: np.ndarray, threshold: float):
        image_resized = cv2.resize(
            image,
            (self.width, self.height),
            interpolation=cv2.INTER_LINEAR,
        )
        darknet_image = self.darknet.make_image(self.width, self.height, 3)
        self.darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = self.darknet.detect_image(
            self.network,
            self.class_names,
            darknet_image,
            thresh=self.prediction_threshold,
        )
        self.darknet.free_image(darknet_image)

        if self.verbose:
            self.darknet.print_detections(detections, True)

        scale_x = image.shape[1] / self.width
        scale_y = image.shape[0] / self.height
        result = Detection2D(source_img=image, boxes_xyxy=[], masks=None)
        for label, confidence, (cx, cy, w, h) in detections:
            x1 = (cx - 0.5 * w) * scale_x
            y1 = (cy - 0.5 * h) * scale_y
            x2 = (cx + 0.5 * w) * scale_x
            y2 = (cy + 0.5 * h) * scale_y
            result.add_detection(
                id=label,
                score=float(confidence),
                box=np.array([x1, y1, x2, y2], dtype=np.float32),
                mask=None,
            )
        return result


class DetrBackend(BaseInferenceBackend):
    def __init__(
        self,
        model="facebook/detr-resnet-50",
        revision="no_timm",
        logger=None,
        **_kwargs,
    ):
        try:
            from transformers import DetrForObjectDetection, DetrImageProcessor
        except ImportError as e:
            raise RuntimeError(
                "DetrBackend requires the transformers package"
            ) from e

        self._logger = logger
        self.processor = DetrImageProcessor.from_pretrained(model, revision=revision)
        self.model = DetrForObjectDetection.from_pretrained(model, revision=revision)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self._logger is not None:
            self._logger.info(f"Model on device: {self.device}")

    def infer(self, image: np.ndarray, threshold=0.7):
        pil_image = Image.fromarray(image)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=threshold,
        )[0]

        det2d = Detection2D(source_img=image, boxes_xyxy=[], masks=None)
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            box_np = box.detach().cpu().numpy().astype(np.float32)
            label_name = self.model.config.id2label[int(label)]
            conf = float(score)
            if self._logger is not None:
                self._logger.info(f"Detected {label_name} {conf:.3f} @ {box_np.tolist()}")
            det2d.add_detection(
                id=label_name,
                score=conf,
                box=box_np,
                mask=None,
            )

        return det2d


class FastSamBackend(BaseInferenceBackend):
    def __init__(
        self,
        model: str = "FastSAM-s.pt",
        logger=None,
        use_ram_labels: bool = False,
        ram_model_path: str = "",
        ram_image_size: int = 384,
        ram_batch_size: int = 8,
        **_kwargs,
    ):
        try:
            from ultralytics import FastSAM
        except ImportError as e:
            raise RuntimeError(
                "FastSamBackend requires the 'ultralytics' package with FastSAM support"
            ) from e

        self._logger = logger
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = FastSAM(model)
        self.use_ram_labels = use_ram_labels
        self.ram_batch_size = max(1, int(ram_batch_size))
        self._ram = None
        if self.use_ram_labels:
            self._ram = _RamLabeler(
                model_path=ram_model_path,
                image_size=ram_image_size,
                device=self.device,
                logger=logger,
            )

    def infer(self, image: np.ndarray, threshold=0.5, iou=0.9):
        prediction = self.model(
            image,
            conf=threshold,
            iou=iou,
            retina_masks=True,
            verbose=False,
            device=self.device,
        )[0]

        boxes = getattr(prediction, "boxes", None)
        masks = getattr(prediction, "masks", None)
        mask_data = None if masks is None or masks.data is None else masks.data.detach().cpu().numpy()

        det2d = Detection2D(
            source_img=image,
            boxes_xyxy=[],
            masks=[] if mask_data is not None else None,
        )
        if boxes is None or boxes.xyxy is None:
            return det2d

        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
        classes = None if boxes.cls is None else boxes.cls.detach().cpu().numpy().astype(np.int32)
        names = getattr(prediction, "names", {}) or {}
        ram_labels = {}

        if self._ram is not None:
            crops = []
            crop_indices = []
            for idx, box in enumerate(xyxy):
                if idx >= len(confs) or float(confs[idx]) < threshold:
                    continue
                crop = _crop_image(image, box)
                if crop is None:
                    continue
                crops.append(crop)
                crop_indices.append(idx)
            if crops:
                batch_labels = self._ram.label_images(crops, batch_size=self.ram_batch_size)
                ram_labels = dict(zip(crop_indices, batch_labels))

        for idx, box in enumerate(xyxy):
            class_id = int(classes[idx]) if classes is not None and idx < len(classes) else idx
            full_label = ram_labels.get(idx, names.get(class_id, str(class_id)))
            label_id = _primary_label(full_label)
            mask = None
            if mask_data is not None and idx < len(mask_data):
                mask = mask_data[idx] > 0.5
            det2d.add_detection(
                id=label_id,
                score=float(confs[idx]),
                box=box,
                mask=mask,
                label=full_label,
            )

        return det2d


def _crop_image(image: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
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


def _primary_label(label: Any) -> str:
    text = str(label).strip()
    if not text:
        return "unknown"
    if "|" in text:
        for token in text.split("|"):
            token = token.strip()
            if token:
                return token
    return text


class _RamLabeler:
    def __init__(
        self,
        model_path: str,
        image_size: int,
        device: str,
        logger=None,
    ):
        if not model_path:
            raise ValueError("RAM labeling requires a non-empty 'ram_model_path'.")
        try:
            from ram import get_transform, inference_ram
            from ram.models import ram_plus
        except ImportError as exc:
            raise RuntimeError(
                "RAM labeling requires the 'ram' package from Recognize Anything."
            ) from exc

        self._logger = logger
        self.device = device
        self.transform = get_transform(image_size=image_size)
        self.inference = inference_ram
        self.model = ram_plus(pretrained=model_path, image_size=image_size, vit="swin_l")
        self.model.eval()
        self.model = self.model.to(self.device)
        if self._logger is not None:
            self._logger.info(f"Loaded RAM model on device: {self.device}")

    def label_images(self, images: list[np.ndarray], batch_size: int) -> list[str]:
        if not images:
            return []

        outputs = []
        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            tensors = []
            for image in batch:
                pil_image = Image.fromarray(image)
                tensors.append(self.transform(pil_image))
            batch_tensor = torch.stack(tensors, dim=0).to(self.device)

            with torch.no_grad():
                raw_output, _ = self.inference(batch_tensor, self.model)

            outputs.extend(self._normalize_labels(raw_output, len(batch)))

        return outputs

    def _normalize_labels(self, raw_output: Any, expected_count: int) -> list[str]:
        if isinstance(raw_output, tuple):
            raw_output = raw_output[0]

        if isinstance(raw_output, str):
            return [self._clean_label(raw_output)] * expected_count

        if isinstance(raw_output, list):
            labels = [self._clean_label(item) for item in raw_output]
            if len(labels) == expected_count:
                return labels
            if len(labels) == 1:
                return labels * expected_count

        raise RuntimeError("Unexpected RAM output format while labeling detections.")

    @staticmethod
    def _clean_label(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(item).strip() for item in value if str(item).strip())
        label = str(value).strip()
        return label if label else "unknown"
