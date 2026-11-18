"""Run pretrained YOLO11n person detection on a configured MP4 video."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

from configs.settings import AppSettings, load_settings
from src.models.yolo11n_person_detector import YOLO11nPersonDetector


@dataclass(frozen=True, slots=True)
class YoloVideoRunSummary:
    model_id: str
    weights: str
    source_video: str
    output_directory: str
    output_video: str | None
    frames_processed: int
    detections_total: int
    average_detections_per_frame: float
    model_load_ms: float
    elapsed_seconds: float
    end_to_end_fps: float
    average_preprocess_ms: float
    average_inference_ms: float
    average_postprocess_ms: float
    inference_fps: float
    gpu_name: str
    gpu_peak_allocated_mib: float
    gpu_peak_reserved_mib: float
    report_path: str


@dataclass(slots=True)
class YOLO11nVideoApplication:
    """Own the standalone YOLO11n video-validation lifecycle."""

    settings: AppSettings

    def run(self, source_override: str | Path | None = None) -> YoloVideoRunSummary:
        yolo = self.settings.yolo
        source = self.settings.paths.resolve(source_override or yolo.source)
        if not source.is_file():
            raise FileNotFoundError(
                f"Video not found: {source}. Place the test video at {yolo.source} "
                "or pass --source."
            )

        detector = YOLO11nPersonDetector(yolo)
        model_load_ms = detector.load()
        torch = detector.torch
        torch.cuda.reset_peak_memory_stats(yolo.gpu_id)

        stage_totals = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
        frames_processed = 0
        detections_total = 0
        started = time.perf_counter()

        for result in detector.predict_video(source):
            frames_processed += 1
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                detections_total += len(boxes)

            speed: dict[str, Any] = getattr(result, "speed", {}) or {}
            for stage in stage_totals:
                stage_totals[stage] += float(speed.get(stage, 0.0))

        detector.synchronize()
        elapsed_seconds = time.perf_counter() - started
        if frames_processed == 0:
            raise RuntimeError("YOLO11n completed without processing any video frames")

        output_directory = yolo.output_directory / yolo.run_name
        output_directory.mkdir(parents=True, exist_ok=True)
        video_suffixes = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
        output_video = next(
            (
                path for path in output_directory.iterdir()
                if path.is_file()
                and path.stem == source.stem
                and path.suffix.lower() in video_suffixes
            ),
            None,
        )
        if yolo.save_video and output_video is None:
            raise RuntimeError(
                "Inference completed, but the annotated video was not found in "
                f"{output_directory}"
            )

        average_preprocess_ms = stage_totals["preprocess"] / frames_processed
        average_inference_ms = stage_totals["inference"] / frames_processed
        average_postprocess_ms = stage_totals["postprocess"] / frames_processed
        inference_fps = (
            1000.0 / average_inference_ms if average_inference_ms > 0 else 0.0
        )
        peak_allocated, peak_reserved = detector.peak_memory_mib()
        report_path = output_directory / "performance.json"

        summary = YoloVideoRunSummary(
            model_id=yolo.model_id,
            weights=str(yolo.weights),
            source_video=str(source),
            output_directory=str(output_directory),
            output_video=str(output_video) if output_video else None,
            frames_processed=frames_processed,
            detections_total=detections_total,
            average_detections_per_frame=detections_total / frames_processed,
            model_load_ms=model_load_ms,
            elapsed_seconds=elapsed_seconds,
            end_to_end_fps=frames_processed / elapsed_seconds if elapsed_seconds else 0.0,
            average_preprocess_ms=average_preprocess_ms,
            average_inference_ms=average_inference_ms,
            average_postprocess_ms=average_postprocess_ms,
            inference_fps=inference_fps,
            gpu_name=detector.gpu_name,
            gpu_peak_allocated_mib=peak_allocated,
            gpu_peak_reserved_mib=peak_reserved,
            report_path=str(report_path),
        )
        report_path.write_text(
            json.dumps(asdict(summary), indent=2),
            encoding="utf-8",
        )

        print("=" * 72)
        print("YOLO11n PERSON-DETECTION VIDEO TEST")
        print("=" * 72)
        print(json.dumps(asdict(summary), indent=2))
        print("=" * 72)
        print("YOLO11n VIDEO INFERENCE PASSED")
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default=None,
        help="Optional video path override; defaults to configs/config.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    YOLO11nVideoApplication(load_settings()).run(args.source)


if __name__ == "__main__":
    main()
