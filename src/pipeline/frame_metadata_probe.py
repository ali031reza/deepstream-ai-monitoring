"""Lightweight DeepStream batch-metadata observer."""
from __future__ import annotations

import threading
import time
from typing import Any

from pyservicemaker import BatchMetadataOperator


def _int_field(frame_meta: Any, name: str) -> int | None:
    try:
        value = getattr(frame_meta, name)
        return int(value) if value is not None else None
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


class FrameMetadataProbe(BatchMetadataOperator):
    """Observe frame metadata without copying or retaining pixel buffers.

    Attach this operator to the DeepStream batch stage. BufferRetriever
    independently retrieves the image tensors downstream; both callbacks
    observe the same batched frame stream.
    """

    SAMPLE_LIMIT = 5

    def __init__(self) -> None:
        super().__init__()
        self.total_frames = 0
        self.first_observed_ns: int | None = None
        self.last_observed_ns: int | None = None
        self.sample_frames: list[dict[str, int | None]] = []
        self._lock = threading.Lock()

    def handle_metadata(self, batch_meta) -> None:
        observed_at_unix_ns = time.time_ns()
        observed_at_monotonic_ns = time.monotonic_ns()

        with self._lock:
            for frame_meta in batch_meta.frame_items:
                self.total_frames += 1
                if self.first_observed_ns is None:
                    self.first_observed_ns = observed_at_monotonic_ns
                self.last_observed_ns = observed_at_monotonic_ns

                if len(self.sample_frames) < self.SAMPLE_LIMIT:
                    self.sample_frames.append({
                        "frame_number": _int_field(frame_meta, "frame_number"),
                        "source_id": _int_field(frame_meta, "source_id"),
                        "pad_index": _int_field(frame_meta, "pad_index"),
                        "batch_id": _int_field(frame_meta, "batch_id"),
                        "buffer_pts": _int_field(frame_meta, "buffer_pts"),
                        "observed_at_unix_ns": observed_at_unix_ns,
                        "observed_at_monotonic_ns": observed_at_monotonic_ns,
                        "source_width": _int_field(frame_meta, "source_width"),
                        "source_height": _int_field(frame_meta, "source_height"),
                        "pipeline_width": _int_field(frame_meta, "pipeline_width"),
                        "pipeline_height": _int_field(frame_meta, "pipeline_height"),
                    })

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            elapsed_ns = 0
            if self.first_observed_ns is not None and self.last_observed_ns is not None:
                elapsed_ns = max(0, self.last_observed_ns - self.first_observed_ns)
            elapsed_seconds = elapsed_ns / 1_000_000_000
            observed_fps = (
                (self.total_frames - 1) / elapsed_seconds
                if self.total_frames > 1 and elapsed_seconds > 0
                else 0.0
            )
            return {
                "total_frames": self.total_frames,
                "elapsed_seconds": elapsed_seconds,
                "observed_fps": observed_fps,
                "sample_frames": list(self.sample_frames),
            }
