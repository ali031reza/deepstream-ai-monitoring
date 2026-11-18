"""YOLO11n TensorRT engine adapter for the DeepStream/CuPy runtime.

DeepStream frames arrive as RGB HxWx3 uint8 CuPy arrays. The selected frame is
copied to host memory for PIL letterboxing, then copied into a TensorRT input
binding. TensorRT executes the engine on the selected CUDA GPU. No PyTorch,
Ultralytics or OpenCV import is needed in the DeepStream container.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class YoloDetection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    bbox_normalized: tuple[float, float, float, float]

    def to_payload(self) -> dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
            "bbox_normalized": list(self.bbox_normalized),
            "synthetic": False,
        }


@dataclass(frozen=True, slots=True)
class TensorRTFrameInference:
    frame_rgb: np.ndarray
    detections: tuple[YoloDetection, ...]
    preprocess_ms: float
    host_to_device_ms: float
    inference_ms: float
    postprocess_ms: float


@dataclass(slots=True)
class TensorRTYOLO11nEngine:
    engine_path: str
    gpu_id: int
    image_size: int
    confidence: float
    iou: float
    classes: tuple[int, ...]
    max_detections: int
    class_names: dict[int, str] = field(default_factory=lambda: {0: "person"})
    _cp: Any = field(default=None, init=False, repr=False)
    _trt: Any = field(default=None, init=False, repr=False)
    _runtime: Any = field(default=None, init=False, repr=False)
    _engine: Any = field(default=None, init=False, repr=False)
    _context: Any = field(default=None, init=False, repr=False)
    _stream: Any = field(default=None, init=False, repr=False)
    _input_gpu: Any = field(default=None, init=False, repr=False)
    _output_gpu: Any = field(default=None, init=False, repr=False)
    _input_name: str = field(default="", init=False)
    _output_name: str = field(default="", init=False)
    _input_shape: tuple[int, ...] = field(default=(), init=False)
    _output_shape: tuple[int, ...] = field(default=(), init=False)
    _input_dtype: Any = field(default=None, init=False, repr=False)
    _output_dtype: Any = field(default=None, init=False, repr=False)
    _gpu_name: str = field(default="", init=False)
    _load_ms: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        if self.gpu_id < 0 or self.image_size <= 0 or self.max_detections <= 0:
            raise ValueError("GPU ID, image size and max_detections are invalid")
        if not 0.0 <= self.confidence <= 1.0 or not 0.0 <= self.iou <= 1.0:
            raise ValueError("confidence and IoU must be in [0, 1]")
        if not self.classes:
            raise ValueError("At least one YOLO class must be configured")

    def load(self) -> float:
        if self._context is not None:
            return self._load_ms

        try:
            import cupy as cp
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT runtime is unavailable. Install the DeepStream 9.2 "
                "TensorRT 10.16 Python bindings and ensure CuPy is installed."
            ) from exc

        from pathlib import Path

        engine_path = Path(self.engine_path)
        if not engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        device_count = cp.cuda.runtime.getDeviceCount()
        if self.gpu_id >= device_count:
            raise RuntimeError(
                f"GPU {self.gpu_id} is unavailable; CuPy sees {device_count} device(s)"
            )

        started = perf_counter()
        self._cp = cp
        self._trt = trt

        with cp.cuda.Device(self.gpu_id):
            properties = cp.cuda.runtime.getDeviceProperties(self.gpu_id)
            self._gpu_name = properties["name"]
            if isinstance(self._gpu_name, bytes):
                self._gpu_name = self._gpu_name.decode("utf-8", errors="replace")

            self._stream = cp.cuda.Stream(non_blocking=True)
            logger = trt.Logger(trt.Logger.WARNING)
            self._runtime = trt.Runtime(logger)
            self._engine = self._runtime.deserialize_cuda_engine(
                engine_path.read_bytes()
            )
            if self._engine is None:
                raise RuntimeError(
                    "TensorRT failed to deserialize the engine. Rebuild it with "
                    "the same DeepStream/TensorRT runtime and target GPU."
                )
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError("TensorRT could not create an execution context")

            inputs: list[str] = []
            outputs: list[str] = []
            for index in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(index)
                mode = self._engine.get_tensor_mode(name)
                if mode == trt.TensorIOMode.INPUT:
                    inputs.append(name)
                elif mode == trt.TensorIOMode.OUTPUT:
                    outputs.append(name)

            if len(inputs) != 1 or len(outputs) != 1:
                raise RuntimeError(
                    f"Expected one input and one output; found {len(inputs)} "
                    f"input(s) and {len(outputs)} output(s)"
                )

            self._input_name, self._output_name = inputs[0], outputs[0]
            self._input_shape = tuple(
                int(value) for value in self._engine.get_tensor_shape(self._input_name)
            )
            self._output_shape = tuple(
                int(value) for value in self._engine.get_tensor_shape(self._output_name)
            )
            expected_input = (1, 3, self.image_size, self.image_size)
            if self._input_shape != expected_input:
                raise RuntimeError(
                    f"Engine input is {self._input_shape}; expected {expected_input}. "
                    "Export a static batch-1 ONNX model at the configured image size."
                )
            if any(value <= 0 for value in self._output_shape):
                raise RuntimeError(
                    f"Engine output must have static positive dimensions: {self._output_shape}"
                )
            if len(self._output_shape) != 3 or self._output_shape[0] != 1:
                raise RuntimeError(
                    f"Unexpected YOLO output shape: {self._output_shape}"
                )

            self._input_dtype = np.dtype(
                trt.nptype(self._engine.get_tensor_dtype(self._input_name))
            )
            self._output_dtype = np.dtype(
                trt.nptype(self._engine.get_tensor_dtype(self._output_name))
            )
            if self._input_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
                raise RuntimeError(
                    f"Unsupported TensorRT input dtype: {self._input_dtype}"
                )
            if self._output_dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
                raise RuntimeError(
                    f"Unsupported TensorRT output dtype: {self._output_dtype}"
                )

            self._input_gpu = cp.empty(self._input_shape, dtype=self._input_dtype)
            self._output_gpu = cp.empty(self._output_shape, dtype=self._output_dtype)

            if not self._context.set_tensor_address(
                self._input_name, int(self._input_gpu.data.ptr)
            ):
                raise RuntimeError("Could not bind TensorRT input memory")
            if not self._context.set_tensor_address(
                self._output_name, int(self._output_gpu.data.ptr)
            ):
                raise RuntimeError("Could not bind TensorRT output memory")

            self._stream.synchronize()

        self._load_ms = (perf_counter() - started) * 1000.0
        return self._load_ms

    @property
    def gpu_name(self) -> str:
        if not self._gpu_name:
            raise RuntimeError("Load the TensorRT engine first")
        return self._gpu_name

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape

    @property
    def load_ms(self) -> float:
        return self._load_ms

    def infer(self, frame_gpu: Any) -> TensorRTFrameInference:
        if self._context is None:
            self.load()
        cp = self._cp

        if not isinstance(frame_gpu, cp.ndarray):
            raise TypeError("DeepStream frame must be a CuPy array")
        if frame_gpu.device.id != self.gpu_id:
            raise ValueError(
                f"Frame is on GPU {frame_gpu.device.id}, expected {self.gpu_id}"
            )
        if frame_gpu.dtype != cp.uint8 or frame_gpu.ndim != 3 or frame_gpu.shape[2] != 3:
            raise ValueError("Expected RGB uint8 frame with shape (H, W, 3)")

        preprocess_started = perf_counter()
        frame_rgb = cp.asnumpy(frame_gpu)
        original_height, original_width = frame_rgb.shape[:2]

        from PIL import Image

        ratio = min(
            self.image_size / original_width,
            self.image_size / original_height,
        )
        resized_width = max(1, int(round(original_width * ratio)))
        resized_height = max(1, int(round(original_height * ratio)))
        pad_width = self.image_size - resized_width
        pad_height = self.image_size - resized_height
        left = int(round(pad_width / 2.0 - 0.1))
        top = int(round(pad_height / 2.0 - 0.1))

        resized = Image.fromarray(frame_rgb, mode="RGB").resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BILINEAR,
        )
        letterboxed = Image.new(
            "RGB", (self.image_size, self.image_size), (114, 114, 114)
        )
        letterboxed.paste(resized, (left, top))
        input_host = np.asarray(letterboxed, dtype=np.uint8)
        input_host = np.ascontiguousarray(
            input_host.transpose(2, 0, 1)[None],
            dtype=np.float32,
        )
        input_host *= np.float32(1.0 / 255.0)
        input_host = np.ascontiguousarray(input_host.astype(self._input_dtype))
        preprocess_ms = (perf_counter() - preprocess_started) * 1000.0

        with cp.cuda.Device(self.gpu_id):
            transfer_start = cp.cuda.Event()
            transfer_end = cp.cuda.Event()
            inference_start = cp.cuda.Event()
            inference_end = cp.cuda.Event()

            with self._stream:
                transfer_start.record(self._stream)
                self._input_gpu.set(input_host, stream=self._stream)
                transfer_end.record(self._stream)

                inference_start.record(self._stream)
                executed = self._context.execute_async_v3(
                    stream_handle=int(self._stream.ptr)
                )
                if not executed:
                    raise RuntimeError("TensorRT execute_async_v3 returned False")
                inference_end.record(self._stream)

            self._stream.synchronize()
            transfer_ms = float(cp.cuda.get_elapsed_time(
                transfer_start, transfer_end
            ))
            inference_ms = float(cp.cuda.get_elapsed_time(
                inference_start, inference_end
            ))

            postprocess_started = perf_counter()
            output_host = self._output_gpu.get(
                stream=self._stream, blocking=True
            ).astype(np.float32, copy=False)
            detections = self.decode_yolo_output(
                output_host,
                frame_shape=(original_height, original_width),
                ratio=ratio,
                pad_left=left,
                pad_top=top,
                confidence=self.confidence,
                iou=self.iou,
                classes=self.classes,
                max_detections=self.max_detections,
                class_names=self.class_names,
            )
            postprocess_ms = (perf_counter() - postprocess_started) * 1000.0

        return TensorRTFrameInference(
            frame_rgb=frame_rgb,
            detections=tuple(detections),
            preprocess_ms=preprocess_ms,
            host_to_device_ms=transfer_ms,
            inference_ms=inference_ms,
            postprocess_ms=postprocess_ms,
        )

    @staticmethod
    def _nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
        max_detections: int,
    ) -> list[int]:
        if boxes.size == 0:
            return []
        x1, y1, x2, y2 = (
            boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        )
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        kept: list[int] = []

        while order.size and len(kept) < max_detections:
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break
            remaining = order[1:]
            xx1 = np.maximum(x1[current], x1[remaining])
            yy1 = np.maximum(y1[current], y1[remaining])
            xx2 = np.minimum(x2[current], x2[remaining])
            yy2 = np.minimum(y2[current], y2[remaining])
            intersection = (
                np.maximum(0.0, xx2 - xx1)
                * np.maximum(0.0, yy2 - yy1)
            )
            union = areas[current] + areas[remaining] - intersection + 1e-7
            ious = intersection / union
            order = remaining[ious <= iou_threshold]

        return kept

    @classmethod
    def decode_yolo_output(
        cls,
        output: np.ndarray,
        *,
        frame_shape: tuple[int, int],
        ratio: float,
        pad_left: int,
        pad_top: int,
        confidence: float,
        iou: float,
        classes: tuple[int, ...],
        max_detections: int,
        class_names: dict[int, str] | None = None,
    ) -> list[YoloDetection]:
        """Decode YOLO11 output in [1, 84, 8400] or [1, 8400, 84] layout."""
        predictions = np.asarray(output, dtype=np.float32)
        if predictions.ndim == 3:
            if predictions.shape[0] != 1:
                raise ValueError(
                    f"Only batch-1 YOLO output is supported: {predictions.shape}"
                )
            predictions = predictions[0]
        if predictions.ndim != 2:
            raise ValueError(f"Unexpected YOLO output rank: {predictions.ndim}")

        # Normalize to [candidate_count, 4 + class_count].
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T
        if predictions.shape[1] < 5:
            raise ValueError(f"Unexpected YOLO output shape: {predictions.shape}")

        class_scores = predictions[:, 4:]
        if class_scores.shape[1] == 0:
            return []
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        class_mask = np.isin(class_ids, np.asarray(classes, dtype=np.int64))
        candidate_mask = (scores >= confidence) & class_mask
        if not np.any(candidate_mask):
            return []

        selected = predictions[candidate_mask]
        selected_scores = scores[candidate_mask]
        selected_class_ids = class_ids[candidate_mask]
        center_x, center_y, box_width, box_height = (
            selected[:, 0], selected[:, 1], selected[:, 2], selected[:, 3]
        )
        boxes = np.column_stack((
            center_x - box_width / 2.0,
            center_y - box_height / 2.0,
            center_x + box_width / 2.0,
            center_y + box_height / 2.0,
        )).astype(np.float32)

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / ratio
        height, width = frame_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)

        names = class_names or {0: "person"}
        detections: list[YoloDetection] = []
        for class_id in classes:
            class_indices = np.flatnonzero(selected_class_ids == class_id)
            if class_indices.size == 0:
                continue
            class_boxes = boxes[class_indices]
            class_confidences = selected_scores[class_indices]
            keep = cls._nms(
                class_boxes, class_confidences, iou, max_detections
            )
            for local_index in keep:
                x1, y1, x2, y2 = map(float, class_boxes[local_index])
                if x2 <= x1 or y2 <= y1:
                    continue
                detection_confidence = float(class_confidences[local_index])
                detections.append(YoloDetection(
                    class_id=int(class_id),
                    class_name=names.get(int(class_id), f"class_{class_id}"),
                    confidence=detection_confidence,
                    bbox_xyxy=(x1, y1, x2, y2),
                    bbox_normalized=(
                        x1 / width, y1 / height, x2 / width, y2 / height
                    ),
                ))

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections[:max_detections]

    def close(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()
        self._context = None
        self._engine = None
        self._runtime = None
        self._input_gpu = None
        self._output_gpu = None
        self._stream = None
