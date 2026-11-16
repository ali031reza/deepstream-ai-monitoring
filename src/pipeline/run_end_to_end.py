"""Phase 3: supervised, reproducible DeepStream -> Kafka -> PostgreSQL run.

This starts the Phase-1 results processor in a subprocess, runs the existing
DeepStream shared-frame pipeline with Phase-2 JPEG events, waits for the exact
session's two-model results in PostgreSQL, checks referenced JPEGs, and stops
the consumer cleanly. No Kafka topics/DB data are deleted.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

from src.monitoring.metrics import save_json


def wait_for_session(database, session, expected, store, timeout, consumer=None):
    from src.storage.verify_event_images import verify_session
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        if consumer and consumer.poll() not in (None, 0):
            raise RuntimeError(f"Results processor exited early with code {consumer.returncode}")
        try:
            report = verify_session(database, store, session)
            last = report
            models = report["models"]
            if (
                set(models) == set(expected)
                and all(models[k]["rows"] == n for k, n in expected.items())
                and sum(row["events"] for row in models.values()) >= 0
            ):
                return report
            if any(models.get(k, {}).get("rows", 0) > n for k, n in expected.items()):
                raise RuntimeError(f"Unexpected extra results for {session}: {models}")
        except (OSError, ConnectionError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"PostgreSQL did not reach expected model counts {expected}; last={last}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-id", default="camera_0")
    parser.add_argument("--event-root", default="artifacts/event_frames")
    parser.add_argument("--event-min-confidence", type=float, default=.65)
    parser.add_argument("--event-every-n", type=int, default=30)
    parser.add_argument("--event-class", default=None)
    parser.add_argument("--model-a-workers", type=int, default=1)
    parser.add_argument("--model-b-workers", type=int, default=1)
    parser.add_argument("--model-queue-size", type=int, default=4)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--kafka-config", default="config/kafka.json")
    parser.add_argument("--db-timeout", type=float, default=180)
    parser.add_argument("--metrics-dir", default="artifacts/benchmarks")
    parser.add_argument("--consumer-group", default=None,
                        help="Existing group or a fresh group; new groups use earliest for safety")
    args = parser.parse_args()
    if args.db_timeout <= 0:
        parser.error("--db-timeout must be positive")
    if not Path(args.source).is_file():
        parser.error(f"MP4 not found: {args.source}")
    if not os.environ.get("DATABASE_PASSWORD") and not os.environ.get("POSTGRES_PASSWORD"):
        parser.error("DATABASE_PASSWORD must be provided via environment")
    from src.kafka.runtime_config import load_runtime_config
    from src.database.client import DatabaseClient
    from src.storage.event_frame_store import EventFrameStore
    config = load_runtime_config(args.kafka_config)
    database = DatabaseClient(**config["database"])
    if not database.check_connection():
        raise RuntimeError("PostgreSQL not reachable")
    session = uuid4().hex
    metrics_root = Path(args.metrics_dir)
    worker_report = metrics_root / f"phase234_{session}.json"
    end_to_end_report = metrics_root / f"end_to_end_{session}.json"
    environment = os.environ.copy()
    environment["KAFKA_AUTO_OFFSET_RESET"] = "earliest"
    if args.consumer_group:
        environment["RESULTS_CONSUMER_GROUP_ID"] = args.consumer_group
    else:
        # Existing stable group minimizes replay of old topics on repeated runs.
        environment.setdefault("RESULTS_CONSUMER_GROUP_ID", "deepstream-results-processor")
    processor_cmd = [sys.executable, "-m", "src.results.run_results_processor",
                     "--config", args.kafka_config]
    producer_cmd = [
        sys.executable, "-m", "src.pipeline.run_event_pipeline",
        "--source", args.source, "--source-id", args.source_id,
        "--gpu-id", str(args.gpu_id), "--queue-size", str(args.model_queue_size),
        "--kafka-config", args.kafka_config,
        "--event-root", args.event_root,
        "--event-every-n", str(args.event_every_n),
        "--event-min-confidence", str(args.event_min_confidence),
        "--model-a-workers", str(args.model_a_workers),
        "--model-b-workers", str(args.model_b_workers),
        "--metrics-json", str(worker_report),
        "--session-id", session,
    ]
    if args.event_class:
        producer_cmd += ["--event-class", args.event_class]
    processor = None
    result = None
    try:
        print(f"E2E session={session}, consumer_group={environment['RESULTS_CONSUMER_GROUP_ID']}", flush=True)
        processor = subprocess.Popen(processor_cmd, env=environment)
        # For *fresh* groups earliest prevents startup races; for an existing
        # group already committed offsets are honored, even with earliest.
        time.sleep(2)
        if processor.poll() is not None:
            raise RuntimeError("Results processor failed to start")
        subprocess.run(producer_cmd, env=environment, check=True)
        report = json.loads(worker_report.read_text(encoding="utf-8"))
        counts = report["metrics"]["counts"]
        expected = {
            "mock_model_a": counts.get("mock_model_a.kafka_enqueue", 0),
            "mock_model_b": counts.get("mock_model_b.kafka_enqueue", 0),
        }
        if not expected["mock_model_a"] or expected["mock_model_a"] != expected["mock_model_b"]:
            raise AssertionError(f"Model frame counts differ or empty: {expected}")
        result = wait_for_session(
            database, session, expected, EventFrameStore(args.event_root),
            args.db_timeout, consumer=processor,
        )
        saved = report["event_store_stats"]["saved"]
        if result["verified_unique_images"] != saved:
            raise AssertionError(
                f"PostgreSQL references {result['verified_unique_images']} event images but JPEG store wrote {saved}"
            )
        result.update({"expected": expected, "event_store_saved": saved,
                       "producer_metrics": str(worker_report)})
        save_json(end_to_end_report, result)
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    finally:
        if processor and processor.poll() is None:
            processor.send_signal(signal.SIGINT)
            try:
                processor.wait(timeout=30)
            except subprocess.TimeoutExpired:
                processor.terminate()
                try:
                    processor.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    processor.kill()
                    processor.wait()
        if processor and processor.returncode not in (None, 0):
            print(f"WARNING: results processor exited code={processor.returncode}", file=sys.stderr)
    if processor is None or processor.returncode != 0:
        raise RuntimeError("Results processor did not stop cleanly")
    print("PASS: DeepStream -> 2 GPU mock models -> Kafka -> PostgreSQL + JPEG storage", flush=True)


if __name__ == "__main__":
    main()
