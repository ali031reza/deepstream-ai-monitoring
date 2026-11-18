
"""Convert TensorRT inference output into the backend Protobuf contract."""

from collections.abc import Mapping
from numbers import Integral, Real
from uuid import NAMESPACE_URL, uuid5

import math
import time

from src.proto.inference_result_pb2 import InferenceResult


def _required_string(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _nonnegative_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_int(value, name):
    number = _nonnegative_int(value, name)
    if number == 0:
        raise ValueError(f"{name} must be positive")
    return number


def _finite_float(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")

    return number


def build_inference_result(
    model_output: Mapping,
    *,
    camera_id: str,
    source_session_id: str,
    frame_sequence: int,
    captured_at_ms: int,
    worker_id: str,
    processed_at_ms: int | None = None,
) -> InferenceResult:

    if not isinstance(model_output, Mapping):
        raise TypeError("model_output must be a mapping")

    camera_id = _required_string(camera_id, "camera_id")
    source_session_id = _required_string(
        source_session_id, "source_session_id"
    )
    worker_id = _required_string(worker_id, "worker_id")

    frame_sequence = _nonnegative_int(
        frame_sequence, "frame_sequence"
    )
    captured_at_ms = _positive_int(
        captured_at_ms, "captured_at_ms"
    )

    model_id = _required_string(
        model_output.get("model_id"), "model_id"
    )

    model_frame_id = _nonnegative_int(
        model_output.get("frame_id"), "model_output.frame_id"
    )

    if model_frame_id != frame_sequence:
        raise ValueError(
            f"Frame mismatch: model={model_frame_id}, "
            f"source={frame_sequence}"
        )

    shape = (
        model_output.get("input_frame_shape")
        or model_output.get("frame_shape")
    )

    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        raise ValueError("Missing input frame shape")

    frame_height = _positive_int(shape[0], "frame_height")
    frame_width = _positive_int(shape[1], "frame_width")

    gpu_id = _nonnegative_int(
        model_output.get("gpu_id", 0), "gpu_id"
    )

    result = InferenceResult()

    result.camera_id = camera_id
    result.source_session_id = source_session_id
    result.frame_sequence = frame_sequence

    result.frame_id = (
        f"{source_session_id}:{frame_sequence:012d}"
    )

    result.model_id = model_id
    result.worker_id = worker_id
    result.gpu_id = gpu_id

    result.frame_width = frame_width
    result.frame_height = frame_height

    model_input_size = model_output.get("model_input_size")

    if model_input_size is not None:
        result.model_input_size = _positive_int(
            model_input_size, "model_input_size"
        )

    result.schema_version = "2.0"

    # Stable ID for retries of the same model/frame result.
    idempotency_key = (
        f"{camera_id}:{source_session_id}:"
        f"{frame_sequence}:{model_id}"
    )

    result.result_id = str(
        uuid5(NAMESPACE_URL, idempotency_key)
    )

    result.captured_at_ms = captured_at_ms

    result.processed_at_ms = (
        _positive_int(processed_at_ms, "processed_at_ms")
        if processed_at_ms is not None
        else time.time_ns() // 1_000_000
    )

    result.status = _required_string(
        model_output.get("status", "processed"), "status"
    )

    is_event = model_output.get("is_event", False)

    if not isinstance(is_event, bool):
        raise ValueError("is_event must be a boolean")

    result.is_event = is_event

    if is_event:
        result.event_type = _required_string(
            model_output.get("event_type"), "event_type"
        )
        result.storage_key = _required_string(
            model_output.get("storage_key"), "storage_key"
        )
        result.image_format = model_output.get("image_format", "")
        result.image_size_bytes = _nonnegative_int(
            model_output.get("image_size_bytes", 0),
            "image_size_bytes",
        )
        result.image_sha256 = model_output.get(
            "image_sha256", ""
        )

        result.event_storage_root = model_output["event_storage_root"]
        result.boxed_key = model_output["boxed_key"]
        result.boxed_sha256 = model_output["boxed_sha256"]
        result.boxed_bytes = model_output["boxed_bytes"]
        result.source_video_ms = model_output["source_video_ms"]

    # Convert detections.
    detections = model_output.get("detections", [])

    if not isinstance(detections, list):
        raise ValueError("detections must be a list")

    for index, item in enumerate(detections):

        if not isinstance(item, Mapping):
            raise ValueError(
                f"detections[{index}] must be an object"
            )

        detection = result.detections.add()

        detection.class_id = _nonnegative_int(
            item.get("class_id"),
            f"detections[{index}].class_id",
        )

        detection.class_name = _required_string(
            item.get("class_name"),
            f"detections[{index}].class_name",
        )

        confidence = _finite_float(
            item.get("confidence"),
            f"detections[{index}].confidence",
        )

        if not 0 <= confidence <= 1:
            raise ValueError("Invalid detection confidence")

        detection.confidence = confidence

        bbox = item.get("bbox_normalized")

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(
                f"detections[{index}] must have four bbox coordinates"
            )

        coordinates = [
            _finite_float(value, "bbox coordinate")
            for value in bbox
        ]

        if any(value < 0 or value > 1 for value in coordinates):
            raise ValueError("Bounding-box coordinates must be normalized")

        x1, y1, x2, y2 = coordinates

        if x1 > x2 or y1 > y2:
            raise ValueError("Invalid bounding-box coordinate ordering")

        detection.bbox_normalized.extend(coordinates)

        # Preserve the synthetic flag.
        synthetic = item.get("synthetic", False)

        if not isinstance(synthetic, bool):
            raise ValueError("synthetic must be a boolean")

        detection.synthetic = synthetic

    # Convert GPU timings and optional pipeline timings.
    timings = model_output.get("timings_ms", {})
    pipeline_timings = model_output.get(
        "pipeline_timings_ms", {}
    )

    if not isinstance(timings, Mapping):
        raise ValueError("timings_ms must be a mapping")

    if not isinstance(pipeline_timings, Mapping):
        raise ValueError("pipeline_timings_ms must be a mapping")

    combined_timings = dict(timings)
    combined_timings.update(pipeline_timings)

    for stage, duration in sorted(combined_timings.items()):

        stage = _required_string(stage, "timing stage")
        duration = _finite_float(
            duration, f"timing {stage}"
        )

        if duration < 0:
            raise ValueError("Timing duration cannot be negative")

        result.timings.add(
            stage=stage,
            duration_ms=duration,
        )

    return result
