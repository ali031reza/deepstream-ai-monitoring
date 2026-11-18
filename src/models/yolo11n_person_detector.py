"""Ultralytics-backed YOLO11n person detector.

This adapter intentionally owns the optional PyTorch/Ultralytics dependency.
Importing application configuration or CPU-only tests does not require
either package to be installed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator

from configs.settings import YoloSettings


@dataclass(slots=True)
class YOLO11nPersonDetector:
    """Load pretrained YOLO11n and expose GPU-backed video prediction."""

    settings: YoloSettings
    _torch: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _gpu_name: str | None = field(default=None, init=False)
    _load_time_ms: float = field(default=0.0, init=False)

    def load(self) -> float:
        """Load weights and validate that the configured CUDA device exists."""
        if self._model is not None:
            return self._load_time_ms

        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "YOLO dependencies are missing. Activate the yolo11n WSL "
                "environment and install requirements-yolo.txt."
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is unavailable to PyTorch. YOLO11n is configured for GPU inference."
            )
        if self.settings.gpu_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"Configured CUDA device {self.settings.gpu_id} is unavailable; "
                f"PyTorch detected {torch.cuda.device_count()} GPU(s)."
            )

        self.settings.weights.parent.mkdir(parents=True, exist_ok=True)
        started = perf_counter()
        self._model = YOLO(str(self.settings.weights))
        torch.cuda.synchronize(self.settings.gpu_id)
        self._load_time_ms = (perf_counter() - started) * 1000.0
        self._torch = torch
        self._gpu_name = torch.cuda.get_device_name(self.settings.gpu_id)
        return self._load_time_ms

    @property
    def torch(self) -> Any:
        if self._torch is None:
            raise RuntimeError("Call load() before accessing the PyTorch runtime")
        return self._torch

    @property
    def gpu_name(self) -> str:
        if self._gpu_name is None:
            raise RuntimeError("Call load() before accessing the GPU name")
        return self._gpu_name

    @property
    def load_time_ms(self) -> float:
        return self._load_time_ms

    def predict_video(self, source: Path) -> Iterator[Any]:
        """Yield Ultralytics Results while saving an annotated video."""
        if self._model is None:
            self.load()
        if not source.is_file():
            raise FileNotFoundError(f"YOLO source video does not exist: {source}")

        return self._model.predict(
            source=str(source),
            device=self.settings.gpu_id,
            imgsz=self.settings.image_size,
            conf=self.settings.confidence,
            iou=self.settings.iou,
            classes=list(self.settings.classes),
            quantize=self.settings.quantize,
            max_det=self.settings.max_detections,
            save=self.settings.save_video,
            project=str(self.settings.output_directory),
            name=self.settings.run_name,
            exist_ok=True,
            stream=True,
            verbose=False,
        )

    def synchronize(self) -> None:
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.synchronize(self.settings.gpu_id)

    def peak_memory_mib(self) -> tuple[float, float]:
        """Return peak allocated and reserved CUDA memory in MiB."""
        if self._torch is None:
            return 0.0, 0.0
        allocated = self._torch.cuda.max_memory_allocated(self.settings.gpu_id)
        reserved = self._torch.cuda.max_memory_reserved(self.settings.gpu_id)
        return allocated / (1024**2), reserved / (1024**2)
