"""Versioned JSON envelope for one model's result for one video frame."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


@dataclass(slots=True)
class ModelResultMessage:
    """A self-contained result message; it never contains another model's output."""

    model_id: str
    source_id: str
    frame_id: int
    gpu_id: int
    frame_width: int
    frame_height: int
    detections: list[dict[str, Any]] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    model_input_size: int | None = None
    processed_at_utc: str = field(default_factory=_utc_now)
    message_id: str = field(default_factory=lambda: str(uuid4()))
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        self.model_id = _non_empty_string(self.model_id, "model_id")
        self.source_id = _non_empty_string(self.source_id, "source_id")
        self.frame_id = _non_negative_int(self.frame_id, "frame_id")
        self.gpu_id = _non_negative_int(self.gpu_id, "gpu_id")
        self.frame_width = _positive_int(self.frame_width, "frame_width")
        self.frame_height = _positive_int(self.frame_height, "frame_height")

        if self.model_input_size is not None:
            self.model_input_size = _positive_int(
                self.model_input_size, "model_input_size"
            )

        self.message_id = _non_empty_string(self.message_id, "message_id")
        self.processed_at_utc = _non_empty_string(
            self.processed_at_utc, "processed_at_utc"
        )
        if self.schema_version != "1.0":
            raise ValueError("schema_version must be '1.0'")

        if not isinstance(self.detections, list):
            raise ValueError("detections must be a list")
        copied_detections: list[dict[str, Any]] = []
        for index, detection in enumerate(self.detections):
            if not isinstance(detection, Mapping):
                raise ValueError(f"detections[{index}] must be an object")
            copied_detections.append(copy.deepcopy(dict(detection)))
        self.detections = copied_detections

        if not isinstance(self.timings_ms, Mapping):
            raise ValueError("timings_ms must be an object")
        normalized_timings: dict[str, float] = {}
        for name, value in self.timings_ms.items():
            if not isinstance(name, str) or not name:
                raise ValueError("timings_ms keys must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"timings_ms[{name!r}] must be numeric")
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(
                    f"timings_ms[{name!r}] must be finite and non-negative"
                )
            normalized_timings[name] = number
        self.timings_ms = normalized_timings

    @classmethod
    def from_inference_result(
        cls,
        result: Mapping[str, Any],
        *,
        source_id: str,
        model_id: str | None = None,
    ) -> "ModelResultMessage":
        """Adapt the mock-model result dictionaries used by this project."""

        if not isinstance(result, Mapping):
            raise TypeError("result must be a mapping")

        resolved_model_id = model_id or result.get("model_id")
        shape = result.get("input_frame_shape") or result.get("frame_shape")
        if not isinstance(shape, (list, tuple)) or len(shape) < 2:
            raise ValueError(
                "result must contain input_frame_shape=[height, width, ...]"
            )

        frame_height = int(shape[0])
        frame_width = int(shape[1])
        raw_detections = result.get("detections", [])
        raw_timings = result.get("timings_ms", {})

        if not isinstance(raw_detections, list):
            raise ValueError("result['detections'] must be a list")
        if not isinstance(raw_timings, Mapping):
            raise ValueError("result['timings_ms'] must be a mapping")

        return cls(
            model_id=str(resolved_model_id or ""),
            source_id=source_id,
            frame_id=int(result["frame_id"]),
            gpu_id=int(result.get("gpu_id", 0)),
            frame_width=frame_width,
            frame_height=frame_height,
            model_input_size=(
                int(result["model_input_size"])
                if result.get("model_input_size") is not None
                else None
            ),
            detections=copy.deepcopy(raw_detections),
            timings_ms=dict(raw_timings),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "message_id": self.message_id,
            "model_id": self.model_id,
            "source_id": self.source_id,
            "frame_id": self.frame_id,
            "processed_at_utc": self.processed_at_utc,
            "gpu_id": self.gpu_id,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "model_input_size": self.model_input_size,
            "detection_count": len(self.detections),
            "detections": copy.deepcopy(self.detections),
            "timings_ms": dict(self.timings_ms),
        }

    def to_json_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ModelResultMessage":
        if not isinstance(value, Mapping):
            raise TypeError("message value must be an object")
        return cls(
            schema_version=value.get("schema_version", "1.0"),
            message_id=value.get("message_id", str(uuid4())),
            model_id=value["model_id"],
            source_id=value["source_id"],
            frame_id=value["frame_id"],
            processed_at_utc=value.get("processed_at_utc", _utc_now()),
            gpu_id=value.get("gpu_id", 0),
            frame_width=value["frame_width"],
            frame_height=value["frame_height"],
            model_input_size=value.get("model_input_size"),
            detections=list(value.get("detections", [])),
            timings_ms=dict(value.get("timings_ms", {})),
        )

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> "ModelResultMessage":
        return cls.from_dict(json.loads(payload.decode("utf-8")))
