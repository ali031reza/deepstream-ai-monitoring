"""Benchmark independent Model A/B GPU worker counts over the same MP4."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from configs.settings import load_settings
import subprocess
import sys


def parse_profiles(value: str):
    profiles = []
    for token in value.split(","):
        a, sep, b = token.strip().partition("x")
        if not sep or not a.isdigit() or not b.isdigit() or min(int(a), int(b)) < 1:
            raise ValueError(f"Invalid worker profile {token!r}; expected AxB e.g. 2x1")
        profiles.append((int(a), int(b)))
    if not profiles:
        raise ValueError("No worker profiles")
    return profiles


def main():
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=settings.source.resolved_uri(settings.paths))
    parser.add_argument("--source-id", default=settings.source.camera_id)
    parser.add_argument("--profiles", default=','.join(settings.benchmark.worker_profiles))
    parser.add_argument("--gpu-id", type=int, default=settings.gpu_id)
    parser.add_argument("--kafka-config", default=str(settings.paths.config_file))
    parser.add_argument("--event-root", default=str(settings.events.root))
    parser.add_argument("--report-dir", default=str(settings.benchmark.report_directory))
    parser.add_argument("--db-timeout", type=float, default=settings.benchmark.database_timeout_seconds)
    args = parser.parse_args()
    try:
        profiles = parse_profiles(args.profiles)
    except ValueError as exc:
        parser.error(str(exc))
    output_dir = settings.paths.resolve(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_a_workers, model_b_workers in profiles:
        before = set(output_dir.glob("end_to_end_*.json"))
        cmd = [
            sys.executable, "-m", "src.pipeline.run_end_to_end",
            "--source", str(settings.paths.resolve(args.source)), "--source-id", args.source_id,
            "--gpu-id", str(args.gpu_id),
            "--model-a-workers", str(model_a_workers),
            "--model-b-workers", str(model_b_workers),
            "--kafka-config", args.kafka_config,
            "--event-root", args.event_root,
            "--db-timeout", str(args.db_timeout),
            "--metrics-dir", str(output_dir),
        ]
        subprocess.run(cmd, check=True, cwd=settings.paths.root)
        new_reports = set(output_dir.glob("end_to_end_*.json")) - before
        if len(new_reports) != 1:
            raise RuntimeError("Expected exactly one end-to-end metrics report")
        e2e = json.loads(next(iter(new_reports)).read_text(encoding="utf-8"))
        runner = json.loads(Path(e2e["producer_metrics"]).read_text(encoding="utf-8"))
        metrics = runner["metrics"]
        resources = runner["resources"]
        gpu_values = [gpu["utilization_percent"] for sample in resources
                      for gpu in sample.get("gpus", [])]
        rows.append({
            "model_a_workers": model_a_workers, "model_b_workers": model_b_workers,
            "frames_per_model": e2e["expected"]["mock_model_a"],
            "producer_elapsed_s": metrics["elapsed_seconds"],
            "model_a_infer_p95_ms": metrics["latency_ms"].get("model_a.infer", {}).get("p95"),
            "model_b_infer_p95_ms": metrics["latency_ms"].get("model_b.infer", {}).get("p95"),
            "model_a_kafka_enqueue_p95_ms": metrics["latency_ms"].get("mock_model_a.kafka_enqueue", {}).get("p95"),
            "model_b_kafka_enqueue_p95_ms": metrics["latency_ms"].get("mock_model_b.kafka_enqueue", {}).get("p95"),
            "average_gpu_util_pct": round(sum(gpu_values) / len(gpu_values), 2) if gpu_values else None,
            "event_images": e2e["verified_unique_images"],
            "session_id": e2e["source_session_id"],
        })
    outfile = output_dir / f"gpu_scaling_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    with outfile.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"PASS: GPU scaling matrix saved: {outfile.resolve()}")


if __name__ == "__main__":
    main()
