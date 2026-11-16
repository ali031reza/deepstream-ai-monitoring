"""Bounded latency histograms and thread-safe counters for Phase 4."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from configs.paths import ProjectPaths
import threading
import time


def distribution(samples):
    arr = sorted(samples)
    if not arr:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    def pct(p):
        idx = min(len(arr) - 1, max(0, math.ceil(p * len(arr)) - 1))
        return round(arr[idx], 3)
    return {
        "count": len(arr), "mean": round(sum(arr) / len(arr), 3),
        "p50": pct(.5), "p95": pct(.95), "p99": pct(.99),
        "max": round(arr[-1], 3),
    }


@dataclass
class RunMetrics:
    max_samples: int = 50_000
    _start: float = field(default_factory=time.monotonic)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _counts: dict = field(default_factory=lambda: defaultdict(int))
    _samples: dict = field(default_factory=lambda: defaultdict(list))
    _errors: list = field(default_factory=list)

    def add(self, stage: str, *, duration_ms: float | None = None, count: int = 1):
        with self._lock:
            self._counts[stage] += count
            if duration_ms is not None and len(self._samples[stage]) < self.max_samples:
                self._samples[stage].append(float(duration_ms))

    def error(self, value: str):
        with self._lock:
            if len(self._errors) < 30:
                self._errors.append(value)

    def snapshot(self):
        with self._lock:
            counts = dict(self._counts)
            dists = {k: distribution(v) for k, v in self._samples.items()}
            errors = list(self._errors)
        elapsed = max(.000001, time.monotonic() - self._start)
        return {
            "elapsed_seconds": round(elapsed, 3),
            "counts": counts,
            "latency_ms": dists,
            "per_second": {k: round(v / elapsed, 2) for k, v in counts.items()},
            "errors": errors,
        }


def save_json(path: str | Path, document: dict):
    target = ProjectPaths.from_environment().resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target
