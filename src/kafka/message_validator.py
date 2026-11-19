from __future__ import annotations

import re
from math import isfinite
from typing import Iterable

from src.proto.inference_result_pb2 import InferenceResult


SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")
MAX_SIGNED_BIGINT = (2 ** 63) - 1
MAX_SIGNED_INTEGER = (2 ** 31) - 1


class MessageValidationError(ValueError):
    def __init__(self, errors: Iterable[str]):
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def validate_inference_result(
    message: InferenceResult,
    *,
    allowed_model_ids: set[str] | None = None,
    require_schema_version: bool = True,
) -> None:
    if not isinstance(message, InferenceResult):
        raise MessageValidationError([
            "Message must be an InferenceResult Protobuf object"
        ])

    errors = []

    for field in (
        "camera_id",
        "source_session_id",
        "frame_id",
        "model_id",
        "worker_id",
        "result_id",
        "status",
    ):
        if not getattr(message, field).strip():
            errors.append(f"{field} is required")

    if allowed_model_ids is not None and message.model_id not in allowed_model_ids:
        errors.append(f"Unsupported model_id: {message.model_id}")

    if require_schema_version and not message.schema_version.strip():
        errors.append("schema_version is required")

    if message.processed_at_ms <= 0:
        errors.append("processed_at_ms must be positive")
    if message.captured_at_ms <= 0:
        errors.append("captured_at_ms must be positive")

    if message.frame_sequence < 0 or message.frame_sequence > MAX_SIGNED_BIGINT:
        errors.append("frame_sequence is outside PostgreSQL BIGINT range")

    # 0 is the proto3 default for producers that did not send field 29.
    # Those messages stay unversioned. A positive value must fit INTEGER.
    if message.camera_config_version > MAX_SIGNED_INTEGER:
        errors.append("camera_config_version exceeds PostgreSQL INTEGER range")

    if message.gpu_id > MAX_SIGNED_INTEGER:
        errors.append("gpu_id exceeds PostgreSQL INTEGER range")

    for field in ("frame_width", "frame_height", "model_input_size"):
        value = getattr(message, field)
        if value < 0 or value > MAX_SIGNED_INTEGER:
            errors.append(f"{field} is outside PostgreSQL INTEGER range")

    if message.frame_width <= 0:
        errors.append("frame_width must be positive")
    if message.frame_height <= 0:
        errors.append("frame_height must be positive")

    if message.is_event and not message.event_type.strip():
        errors.append("event_type is required for event results")

    image_metadata_present = any((
        message.storage_key,
        message.image_format,
        message.image_size_bytes,
        message.image_sha256,
    ))
    image_metadata_complete = all((
        message.storage_key,
        message.image_format,
        message.image_size_bytes,
        message.image_sha256,
    ))

    if message.is_event and not image_metadata_complete:
        errors.append("Event results require complete image metadata")
    if not message.is_event and image_metadata_present:
        errors.append("Non-event results cannot contain event-image metadata")

    if message.image_size_bytes < 0:
        errors.append("image_size_bytes cannot be negative")

    if message.is_event:
        if not message.event_storage_root.startswith("s3://"):
            errors.append("Event requires an S3 storage root")
        if message.boxed_key != message.storage_key.removesuffix(".jpg") + ".boxed.jpg":
            errors.append("Event requires a matching boxed JPEG key")
        if message.boxed_bytes <= 0 or not SHA256_PATTERN.fullmatch(message.boxed_sha256):
            errors.append("Event requires boxed JPEG size and SHA-256")
        if not isfinite(message.source_video_ms) or message.source_video_ms < 0:
            errors.append("Event requires a nonnegative video timestamp")
        key = message.storage_key
        if (
            not key
            or key.startswith("/")
            or "\\" in key
            or any(part in ("", ".", "..") for part in key.split("/"))
        ):
            errors.append("Invalid event-image storage key")

        if message.image_format.lower() not in ("jpeg", "jpg"):
            errors.append("Unsupported event-image format")
        if message.image_size_bytes <= 0:
            errors.append("image_size_bytes must be positive")
        if not SHA256_PATTERN.fullmatch(message.image_sha256):
            errors.append("image_sha256 must be a 64-character hexadecimal digest")

    for index, detection in enumerate(message.detections):
        prefix = f"detections[{index}]"

        if detection.class_id < 0:
            errors.append(f"{prefix}.class_id cannot be negative")
        if not detection.class_name.strip():
            errors.append(f"{prefix}.class_name is required")

        confidence = detection.confidence
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            errors.append(f"{prefix}.confidence must be between 0 and 1")

        bbox = list(detection.bbox_normalized)
        if len(bbox) != 4:
            errors.append(f"{prefix}.bbox_normalized requires four coordinates")
            continue
        if not all(isfinite(value) for value in bbox):
            errors.append(f"{prefix}.bbox_normalized contains non-finite values")
            continue
        if not all(0.0 <= value <= 1.0 for value in bbox):
            errors.append(f"{prefix}.bbox_normalized must be normalized")
        x1, y1, x2, y2 = bbox
        if x1 >= x2 or y1 >= y2:
            errors.append(f"{prefix} has invalid bounding-box ordering")

    seen_stages = set()
    for index, timing in enumerate(message.timings):
        prefix = f"timings[{index}]"
        if not timing.stage.strip():
            errors.append(f"{prefix}.stage is required")
        if timing.stage in seen_stages:
            errors.append(f"Duplicate timing stage: {timing.stage}")
        seen_stages.add(timing.stage)
        if not isfinite(timing.duration_ms) or timing.duration_ms < 0:
            errors.append(
                f"{prefix}.duration_ms must be finite and non-negative"
            )

    if errors:
        raise MessageValidationError(errors)
