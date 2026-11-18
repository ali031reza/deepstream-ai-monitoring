"""Timestamp-based frame sampling for camera/video streams."""
from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass(frozen=True, slots=True)
class SampleDecision:
    should_process: bool
    source_timestamp_ms: float
    sample_index: int | None


@dataclass(slots=True)
class FrameSampler:
    """Select the first decoded frame at/after each source-time boundary.

    PTS is preferred. If PTS is missing or moves backwards, frame number and
    fallback_source_fps provide a deterministic clock. The origin is reset
    to the first valid timestamp, so a non-zero initial PTS does not cause a
    long catch-up loop.
    """

    target_fps: float = 8.0
    fallback_source_fps: float = 30.0
    _next_target_ms: float = field(default=0.0, init=False)
    _last_timestamp_ms: float | None = field(default=None, init=False)
    _origin_ms: float | None = field(default=None, init=False)
    _origin_frame_number: int | None = field(default=None, init=False)
    _sample_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.target_fps) or self.target_fps <= 0:
            raise ValueError("target_fps must be positive and finite")
        if not math.isfinite(self.fallback_source_fps) or self.fallback_source_fps <= 0:
            raise ValueError("fallback_source_fps must be positive and finite")

    @property
    def interval_ms(self) -> float:
        return 1000.0 / self.target_fps

    @property
    def samples_taken(self) -> int:
        return self._sample_count

    def decide(self, *, frame_number: int, pts_ns: int | None) -> SampleDecision:
        if isinstance(frame_number, bool) or not isinstance(frame_number, int) or frame_number < 0:
            raise ValueError("frame_number must be a nonnegative integer")

        valid_pts = (
            pts_ns is not None
            and not isinstance(pts_ns, bool)
            and 0 <= int(pts_ns) < (2**63 - 1)
        )
        if self._origin_frame_number is None:
            self._origin_frame_number = frame_number

        fallback_timestamp_ms = (self._origin_ms or 0.0) + (
            frame_number - self._origin_frame_number
        ) * 1000.0 / self.fallback_source_fps
        timestamp_ms = (
            int(pts_ns) / 1_000_000.0 if valid_pts else fallback_timestamp_ms
        )

        if (
            self._last_timestamp_ms is not None
            and timestamp_ms <= self._last_timestamp_ms
        ):
            timestamp_ms = fallback_timestamp_ms

        if self._origin_ms is None:
            self._origin_ms = timestamp_ms
            # The fallback clock starts at this first observed frame.
            fallback_timestamp_ms = timestamp_ms
        self._last_timestamp_ms = timestamp_ms

        elapsed_ms = max(0.0, timestamp_ms - self._origin_ms)
        tolerance_ms = 0.001

        if elapsed_ms + tolerance_ms < self._next_target_ms:
            return SampleDecision(False, timestamp_ms, None)

        sample_index = self._sample_count
        self._sample_count += 1

        # If a PTS gap skips multiple target boundaries, emit this decoded
        # frame once and advance to the next future boundary.
        while self._next_target_ms <= elapsed_ms + tolerance_ms:
            self._next_target_ms += self.interval_ms

        return SampleDecision(True, timestamp_ms, sample_index)
