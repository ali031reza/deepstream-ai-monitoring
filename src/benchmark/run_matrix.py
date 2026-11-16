"""Phase 4 workload matrix: compare sustained frame rates / ingestion speed."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import csv
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
from configs.settings import AppSettings, load_settings
import subprocess
import sys


@dataclass(slots=True)
class KafkaLoadMatrix:
    """Own one benchmark matrix invocation and its paths/results."""

    settings: AppSettings
    args: argparse.Namespace

    def run(self) -> None:
        settings = self.settings
        args = self.args
        if args.frames < 1:
            parser.error("--frames must be >= 1")
        rates = [float(x) for x in args.rates.split(",")]
        if not rates or any(x < 0 for x in rates):
            parser.error("--rates must contain nonnegative numbers")
        directory = settings.paths.resolve(args.report_dir)
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for rate in rates:
            before = set(glob.glob(str(directory / "load_*.json")))
            command = [
                sys.executable, "-m", "src.benchmark.kafka_load",
                "--frames", str(args.frames), "--rate", str(rate),
                "--kafka-config", args.kafka_config,
                "--report-dir", str(directory), "--await-database",
                "--db-timeout", str(args.db_timeout),
            ]
            subprocess.run(command, check=True, cwd=settings.paths.root)
            written = set(glob.glob(str(directory / "load_*.json"))) - before
            if len(written) != 1:
                raise RuntimeError("Could not identify this profile's unique result")
            report = json.loads(Path(written.pop()).read_text(encoding="utf-8"))
            rows.append({
                "target_fps": rate, "frames": args.frames, "messages": args.frames * 2,
                "kafka_delivery_seconds": report["kafka_delivery_seconds"],
                "kafka_messages_per_sec": report["kafka_message_throughput_s"],
                "total_including_postgres_sec": report["total_seconds_including_db"],
                "model_a_delivery_p95_ms": (report["metrics"]["latency_ms"].get("mock_model_a.delivered") or {}).get("p95"),
                "model_b_delivery_p95_ms": (report["metrics"]["latency_ms"].get("mock_model_b.delivered") or {}).get("p95"),
                "session_id": report["session_id"],
            })
        output = directory / f"matrix_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
        with output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"PASS: Phase 4 matrix CSV: {output.resolve()}")



def main():
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", default=",".join(str(n) for n in settings.benchmark.target_fps))
    parser.add_argument("--frames", type=int, default=settings.benchmark.load_frames)
    parser.add_argument("--kafka-config", default=str(settings.paths.config_file))
    parser.add_argument("--report-dir", default=str(settings.benchmark.report_directory))
    parser.add_argument("--db-timeout", type=float, default=settings.benchmark.database_timeout_seconds)
    args = parser.parse_args()
    KafkaLoadMatrix(settings, args).run()

if __name__ == "__main__":
    main()
