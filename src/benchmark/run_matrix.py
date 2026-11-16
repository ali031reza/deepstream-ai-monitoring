"""Phase 4 workload matrix: compare sustained frame rates / ingestion speed."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rates", default="25,100,250")
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--kafka-config", default="config/kafka.json")
    parser.add_argument("--report-dir", default="artifacts/benchmarks")
    parser.add_argument("--db-timeout", type=float, default=180.)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be >= 1")
    rates = [float(x) for x in args.rates.split(",")]
    if not rates or any(x < 0 for x in rates):
        parser.error("--rates must contain nonnegative numbers")
    directory = Path(args.report_dir)
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
        subprocess.run(command, check=True)
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


if __name__ == "__main__":
    main()
