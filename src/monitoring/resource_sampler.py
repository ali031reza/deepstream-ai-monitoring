"""Optional local CPU/RAM/GPU resource measurements; never blocks inference."""
from __future__ import annotations
import subprocess
import threading
import time


class ResourceSampler:
    def __init__(self, interval_seconds: float = 2.0, max_samples: int = 10000):
        self.interval_seconds = interval_seconds
        self.max_samples = max_samples
        self._stop = threading.Event()
        self._thread = None
        self._samples = []
        self._lock = threading.Lock()

    def start(self):
        if self._thread is not None:
            raise RuntimeError("ResourceSampler already started")
        self._thread = threading.Thread(target=self._run, name="resource-sampler", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        try:
            import psutil
            process = psutil.Process()
            process.cpu_percent(None)
        except ImportError:
            process = None
        while not self._stop.is_set():
            row = {"timestamp_ms": time.time_ns() // 1_000_000}
            if process:
                try:
                    row["process_cpu_percent"] = process.cpu_percent(None)
                    row["process_rss_mb"] = round(process.memory_info().rss / (1024 * 1024), 2)
                except Exception:
                    pass
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=1.5,
                )
                if result.returncode == 0:
                    gpu_lines = result.stdout.strip().splitlines()
                    row["gpus"] = [
                        {"utilization_percent": int(part.strip().split(',')[0]),
                         "memory_used_mb": int(part.strip().split(',')[1])}
                        for part in gpu_lines if "," in part
                    ]
            except (OSError, subprocess.TimeoutExpired, ValueError):
                pass
            with self._lock:
                if len(self._samples) < self.max_samples:
                    self._samples.append(row)
            self._stop.wait(self.interval_seconds)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=self.interval_seconds + 3)
        with self._lock:
            return list(self._samples)
