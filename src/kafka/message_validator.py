
"""Validation for DeepStream Protobuf inference results."""

from __future__ import annotations

import re
from math import isfinite
from typing import Iterable

from src.proto.inference_result_pb2 import InferenceResult


SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")


class MessageValidationError(ValueError):
    """Raised when an inference result fails validation."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("; ".join(self.errors))


def validate_inference_result(
    message: InferenceResult,
    *,
    allowed_model_ids: set[str] | None = None,
    require_schema_version: bool = True,
) -> None:
    """
    Validate an InferenceResult Protobuf message.

    Raises:
        MessageValidationError: If one or more checks fail.
    """

    if not isinstance(message, InferenceResult):
        raise MessageValidationError(
            ["Message must be an InferenceResult Protobuf object"]
        )

    errors: list[str] = []

    # ---------------------------------------------------------
    # 1. Required identity fields
    # ---------------------------------------------------------

    required_fields = (
        "camera_id",
        "frame_id",
        "model_id",
        "result_id",
        "status",
    )

    for field in required_fields:
        value = getattr(message, field)

        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} is required")

    if allowed_model_ids is not None:
        if message.model_id not in allowed_model_ids:
            errors.append(
                f"Unsupported model_id: {message.model_id}"
            )

    # ---------------------------------------------------------
    # 2. Schema version
    # ---------------------------------------------------------

    if require_schema_version and not message.schema_version.strip():
        errors.append("schema_version is required")

    # ---------------------------------------------------------
    # 3. Timestamps
    # ---------------------------------------------------------

    if message.processed_at_ms <= 0:
        errors.append("processed_at_ms must be positive")

    if message.captured_at_ms < 0:
        errors.append("captured_at_ms cannot be negative")

    # ---------------------------------------------------------
    # 4. Frame dimensions
    # ---------------------------------------------------------

    width = message.frame_width
    height = message.frame_height

    if bool(width) != bool(height):
        errors.append(
            "frame_width and frame_height must be provided together"
        )

    # ---------------------------------------------------------
    # 5. Event and image metadata
    # ---------------------------------------------------------

    if message.is_event and not message.event_type.strip():
        errors.append(
            "event_type is required when is_event is true"
        )

    image_fields = {
        "storage_key": bool(message.storage_key.strip()),
        "image_format": bool(message.image_format.strip()),
        "image_size_bytes": message.image_size_bytes > 0,
        "image_sha256": bool(message.image_sha256.strip()),
    }

    image_metadata_present = any(image_fields.values())
    image_metadata_complete = all(image_fields.values())

    if message.is_event and not image_metadata_complete:
        errors.append(
            "Event results require complete image metadata"
        )

    elif image_metadata_present and not image_metadata_complete:
        errors.append(
            "Image metadata must be either complete or absent"
        )

    if message.image_size_bytes < 0:
        errors.append("image_size_bytes cannot be negative")

    if message.image_sha256:
        if not SHA256_PATTERN.fullmatch(message.image_sha256):
            errors.append(
                "image_sha256 must contain exactly 64 hexadecimal characters"
            )

    # ---------------------------------------------------------
    # 6. Detection validation
    # ---------------------------------------------------------

    for index, detection in enumerate(message.detections):

        prefix = f"detections[{index}]"

        if detection.class_id < 0:
            errors.append(
                f"{prefix}.class_id cannot be negative"
            )

        if not detection.class_name.strip():
            errors.append(
                f"{prefix}.class_name is required"
            )

        confidence = detection.confidence

        if (
            not isfinite(confidence)
            or confidence < 0.0
            or confidence > 1.0
        ):
            errors.append(
                f"{prefix}.confidence must be between 0 and 1"
            )

        bbox = list(detection.bbox_normalized)

        if len(bbox) != 4:
            errors.append(
                f"{prefix}.bbox_normalized must contain exactly 4 values"
            )
            continue

        if not all(isfinite(value) for value in bbox):
            errors.append(
                f"{prefix}.bbox_normalized contains non-finite values"
            )
            continue

        if not all(0.0 <= value <= 1.0 for value in bbox):
            errors.append(
                f"{prefix}.bbox_normalized values must be between 0 and 1"
            )

        x1, y1, x2, y2 = bbox

        if x1 >= x2:
            errors.append(
                f"{prefix}.bounding box must have x1 < x2"
            )

        if y1 >= y2:
            errors.append(
                f"{prefix}.bounding box must have y1 < y2"
            )

    # ---------------------------------------------------------
    # 7. Inference timing validation
    # ---------------------------------------------------------

    for index, timing in enumerate(message.timings):

        prefix = f"timings[{index}]"

        if not timing.stage.strip():
            errors.append(
                f"{prefix}.stage is required"
            )

        if (
            not isfinite(timing.duration_ms)
            or timing.duration_ms < 0
        ):
            errors.append(
                f"{prefix}.duration_ms must be finite and non-negative"
            )

    # ---------------------------------------------------------
    # 8. Final result
    # ---------------------------------------------------------

    if errors:
        raise MessageValidationError(errors)
