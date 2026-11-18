"""Run one end-to-end DeepStream -> Kafka -> PostgreSQL verification session."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from uuid import uuid4

from configs.settings import AppSettings, load_settings
from src.monitoring.metrics import save_json


@dataclass(slots=True)
class SessionAwaiter:
    database: object
    session_id: str
    expected: dict[str, int]
    store: object
    timeout_seconds: float
    processor: subprocess.Popen | None = None

    def wait(self) -> dict:
        from src.storage.verify_event_images import verify_session

        deadline = time.monotonic() + self.timeout_seconds
        last = None
        while time.monotonic() < deadline:
            if self.processor and self.processor.poll() not in (None, 0):
                raise RuntimeError(f"Results processor exited early with code {self.processor.returncode}")
            try:
                report = verify_session(self.database, self.store, self.session_id)
                last = report
                models = report["models"]
                if set(models) == set(self.expected) and all(
                    models[key]["rows"] == count for key, count in self.expected.items()
                ):
                    return report
                if any(models.get(key, {}).get("rows", 0) > count for key, count in self.expected.items()):
                    raise RuntimeError(f"Unexpected extra results for {self.session_id}: {models}")
            except (OSError, ConnectionError):
                pass
            time.sleep(0.5)
        raise TimeoutError(f"PostgreSQL did not reach expected model counts {self.expected}; last={last}")


def wait_for_session(database, session, expected, store, timeout, consumer=None):
    """Compatibility facade used by existing tests and benchmark helpers."""
    return SessionAwaiter(database, session, expected, store, timeout, consumer).wait()


@dataclass(frozen=True, slots=True)
class EndToEndOptions:
    source: Path
    camera_id: str
    event_root: Path
    event_min_confidence: float
    event_every_n: int
    event_class: str | None
    model_a_workers: int
    model_b_workers: int
    model_queue_size: int
    gpu_id: int
    config_file: Path
    db_timeout_seconds: float
    metrics_directory: Path
    consumer_group: str | None


@dataclass(slots=True)
class EndToEndApplication:
    settings: AppSettings
    options: EndToEndOptions

    def _stop_processor(self, processor: subprocess.Popen | None) -> None:
        if processor is None or processor.poll() is not None:
            return
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

    def run(self) -> dict:
        from src.database.client import DatabaseClient
        from src.storage.event_frame_store import EventFrameStore

        opts = self.options
        if not opts.source.is_file():
            raise FileNotFoundError(f"MP4 not found: {opts.source}")
        if opts.db_timeout_seconds <= 0:
            raise ValueError("db_timeout_seconds must be positive")
        database = DatabaseClient(**self.settings.database.connection_kwargs())
        if not database.check_connection():
            raise RuntimeError("PostgreSQL not reachable")

        session = uuid4().hex
        metrics_root = self.settings.paths.ensure_directory(opts.metrics_directory)
        worker_report = metrics_root / f"phase234_{session}.json"
        end_to_end_report = metrics_root / f"end_to_end_{session}.json"
        environment = os.environ.copy()
        environment["KAFKA_AUTO_OFFSET_RESET"] = "earliest"
        environment["RESULTS_CONSUMER_GROUP_ID"] = (
            opts.consumer_group or environment.get("RESULTS_CONSUMER_GROUP_ID", "deepstream-results-processor")
        )
        processor_cmd = [
            sys.executable, "-m", "src.results.run_results_processor",
            "--config", str(opts.config_file),
        ]
        producer_cmd = [
            sys.executable, "-m", "src.pipeline.run_event_pipeline",
            "--source", str(opts.source), "--source-id", opts.camera_id,
            "--gpu-id", str(opts.gpu_id), "--queue-size", str(opts.model_queue_size),
            "--kafka-config", str(opts.config_file),
            "--event-root", str(opts.event_root),
            "--event-every-n", str(opts.event_every_n),
            "--event-min-confidence", str(opts.event_min_confidence),
            "--model-a-workers", str(opts.model_a_workers),
            "--model-b-workers", str(opts.model_b_workers),
            "--metrics-json", str(worker_report),
            "--session-id", session,
        ]
        if opts.event_class:
            producer_cmd += ["--event-class", opts.event_class]

        processor: subprocess.Popen | None = None
        try:
            print(f"E2E session={session} consumer_group={environment['RESULTS_CONSUMER_GROUP_ID']}", flush=True)
            processor = subprocess.Popen(processor_cmd, env=environment, cwd=self.settings.paths.root)
            time.sleep(2)
            if processor.poll() is not None:
                raise RuntimeError("Results processor failed to start")
            subprocess.run(producer_cmd, env=environment, cwd=self.settings.paths.root, check=True)
            report = json.loads(worker_report.read_text(encoding="utf-8"))
            counts = report["metrics"]["counts"]
            expected = {
                "mock_model_a": counts.get("mock_model_a.kafka_enqueue", 0),
                "mock_model_b": counts.get("mock_model_b.kafka_enqueue", 0),
            }
            if not expected["mock_model_a"] or expected["mock_model_a"] != expected["mock_model_b"]:
                raise AssertionError(f"Model frame counts differ or empty: {expected}")
            result = SessionAwaiter(
                database, session, expected, EventFrameStore(opts.event_root),
                opts.db_timeout_seconds, processor,
            ).wait()
            saved = report["event_store_stats"]["saved"]
            if result["verified_unique_images"] != saved:
                raise AssertionError(
                    f"PostgreSQL references {result['verified_unique_images']} event images "
                    f"but JPEG store wrote {saved}"
                )
            result.update({
                "expected": expected, "event_store_saved": saved,
                "producer_metrics": str(worker_report),
            })
            save_json(end_to_end_report, result)
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        finally:
            self._stop_processor(processor)
            if processor and processor.returncode not in (None, 0):
                print(f"WARNING: results processor exited code={processor.returncode}", file=sys.stderr)

        if processor is None or processor.returncode != 0:
            raise RuntimeError("Results processor did not stop cleanly")
        print("PASS: DeepStream -> 2 GPU mock models -> Kafka -> PostgreSQL + JPEG storage", flush=True)
        return result


def build_parser(settings: AppSettings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=settings.source.resolved_uri(settings.paths))
    parser.add_argument("--source-id", default=settings.source.camera_id)
    parser.add_argument("--event-root", default=str(settings.events.root))
    parser.add_argument("--event-min-confidence", type=float, default=settings.events.minimum_confidence)
    parser.add_argument("--event-every-n", type=int, default=settings.events.every_nth_frame)
    parser.add_argument("--event-class", default=settings.events.class_name)
    parser.add_argument("--model-a-workers", type=int, default=settings.models["model_a"].workers)
    parser.add_argument("--model-b-workers", type=int, default=settings.models["model_b"].workers)
    parser.add_argument("--model-queue-size", type=int, default=settings.queues.model_input_capacity)
    parser.add_argument("--gpu-id", type=int, default=settings.gpu_id)
    parser.add_argument("--kafka-config", default=str(settings.paths.config_file))
    parser.add_argument("--db-timeout", type=float, default=settings.benchmark.database_timeout_seconds)
    parser.add_argument("--metrics-dir", default=str(settings.metrics_directory))
    parser.add_argument("--consumer-group", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    settings = load_settings()
    if sum(model.enabled for model in settings.models.values()) != 2:
        raise SystemExit(
            "The PostgreSQL end-to-end runner is still a two-model workflow; "
            "validate the single-model mock pipeline first"
        )
    parser = build_parser(settings)
    args = parser.parse_args(argv)
    if args.db_timeout <= 0 or min(args.model_a_workers, args.model_b_workers, args.model_queue_size) < 1:
        parser.error("timeouts and worker/queue counts must be positive")
    opts = EndToEndOptions(
        source=settings.paths.resolve(args.source),
        camera_id=args.source_id,
        event_root=settings.paths.resolve(args.event_root),
        event_min_confidence=args.event_min_confidence,
        event_every_n=args.event_every_n,
        event_class=args.event_class,
        model_a_workers=args.model_a_workers,
        model_b_workers=args.model_b_workers,
        model_queue_size=args.model_queue_size,
        gpu_id=args.gpu_id,
        config_file=settings.paths.resolve(args.kafka_config),
        db_timeout_seconds=args.db_timeout,
        metrics_directory=settings.paths.resolve(args.metrics_dir),
        consumer_group=args.consumer_group,
    )
    EndToEndApplication(settings, opts).run()


if __name__ == "__main__":
    main()
