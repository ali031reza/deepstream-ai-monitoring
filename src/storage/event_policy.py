"""Deterministic event rule for mock-model outputs (NOT real detection)."""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EventPolicy:
    minimum_confidence: float = 0.65
    every_nth_frame: int = 30
    class_name: str | None = None

    def __post_init__(self):
        if not math.isfinite(self.minimum_confidence) or not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be in [0,1]")
        if self.every_nth_frame < 1:
            raise ValueError("every_nth_frame must be >= 1")

    def matches(self, output: dict, frame_sequence: int) -> bool:
        if frame_sequence < 0 or frame_sequence % self.every_nth_frame:
            return False
        for detection in output.get("detections", ()):
            if not isinstance(detection, dict):
                continue
            if self.class_name and detection.get("class_name") != self.class_name:
                continue
            try:
                score = float(detection.get("confidence", -1))
            except (TypeError, ValueError):
                continue
            if math.isfinite(score) and score >= self.minimum_confidence:
                return True
        return False
