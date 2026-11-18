"""Run the complete DeepStream -> TensorRT -> Kafka -> PostgreSQL path.

The producer remains responsible for decode, frame sampling, TensorRT inference,
event JPEG/JSON persistence and Protobuf publication. This supervisor starts the
existing results processor first, waits until its Kafka group is assigned, runs
the producer, verifies the exact session in PostgreSQL, then shuts the processor
down gracefully.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import UUID, uuid4

import psycopg

from configs.settings import load_settings
from src.database.client import DatabaseClient
from src.kafka.topic_manager import KafkaTopicManager
from src.results.kafka_config import load_kafka_config, topic_for_model
from src.storage.event_frame_store import EventFrameStore
from src.storage.verify_event_images import verify_session


def _option_value(arguments: list[str], flag: str, default: str | None = None) -> str | None:
    for index, value in enumerate(arguments):
        if value == flag and index + 1 < len(arguments):
            return arguments[index + 1]
        if value.startswith(flag + "="):
            return value.split("=", 1)[1]
    return default


def _without_option(arguments: list[str], flag: str) -> list[str]:
    result: list[str] = []
    skip_value = False
    for value in arguments:
        if skip_value:
            skip_value = False
            continue
        if value == flag:
            skip_value = True
            continue
        if value.startswith(flag + "="):
            continue
        result.append(value)
    return result


def _validate_database_schema(database: DatabaseClient) -> None:
    """Fail before decoding video if the required Phase-1/DeepStream schema is absent."""
    try:
        with database.transaction() as connection:
            with connection.cursor() as cursor:
                cursor.execute("""
                    SELECT camera_id, frame_id, model_id, processed_at_ms, status,
                           is_event, event_type, event_image_id, source_session_id,
                           result_id, worker_id, frame_sequence, gpu_id, frame_width,
                           frame_height, model_input_size, captured_at_ms,
                           schema_version, payload_sha256, detections, timings
                    FROM inference_results
                    LIMIT 0
                """)
                cursor.execute("""
                    SELECT id, camera_id, frame_id, storage_key, image_format,
                           file_size_bytes, sha256, captured_at_ms
                    FROM event_images
                    LIMIT 0
                """)
    except Exception as exc:
        raise RuntimeError(
            "PostgreSQL is reachable, but its schema is not ready for DeepStream "
            "results. Apply the existing base schema and "
            "infra/postgres/migrations/003_deepstream_result_metadata.sql."
        ) from exc


def wait_for_consumer_group(
    bootstrap_servers: str,
    group_id: str,
    timeout_seconds: float,
    processor: subprocess.Popen,
) -> dict:
    """Wait until the child results processor has a stable Kafka assignment."""
    from confluent_kafka.admin import AdminClient

    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    deadline = time.monotonic() + timeout_seconds
    last_state = "group not visible"

    while time.monotonic() < deadline:
        return_code = processor.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Results processor exited before Kafka assignment (code={return_code})"
            )

        try:
            futures = admin.describe_consumer_groups(
                [group_id],
                request_timeout=5.0,
            )
            description = futures[group_id].result(timeout=5.0)
            members = getattr(description, "members", ()) or ()
            state = getattr(description, "state", None)
            state_name = str(getattr(state, "name", state)).casefold()
            last_state = f"state={state_name or 'unknown'}, members={len(members)}"

            if members and (state is None or "stable" in state_name):
                print(f"[KAFKA] results processor ready: {last_state}", flush=True)
                return {"state": state_name or "unknown", "members": len(members)}
        except Exception as exc:
            last_state = f"{type(exc).__name__}: {exc}"

        time.sleep(0.25)

    raise TimeoutError(
        f"Results processor did not become ready within {timeout_seconds:.1f}s; "
        f"last Kafka group status: {last_state}"
    )


def wait_for_database_session(
    *,
    database: DatabaseClient,
    store: EventFrameStore,
    session_id: str,
    model_id: str,
    expected_rows: int,
    expected_events: int,
    expected_images: int,
    timeout_seconds: float,
    processor: subprocess.Popen,
) -> dict:
    deadline = time.monotonic() + timeout_seconds
    last_report: dict | None = None
    last_database_error: str | None = None

    while time.monotonic() < deadline:
        return_code = processor.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Results processor exited before PostgreSQL reached the expected "
                f"counts (code={return_code}); last report={last_report}"
            )

        try:
            report = verify_session(database, store, session_id)
            last_report = report
            last_database_error = None
        except (psycopg.OperationalError, psycopg.InterfaceError) as exc:
            last_database_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5)
            continue

        models = report["models"]
        unexpected_models = set(models) - {model_id}
        if unexpected_models:
            raise AssertionError(
                f"Unexpected model results for session {session_id}: "
                f"{sorted(unexpected_models)}"
            )

        model_report = models.get(model_id, {"rows": 0, "events": 0})
        actual_rows = int(model_report["rows"])
        actual_events = int(model_report["events"])

        if actual_rows > expected_rows:
            raise AssertionError(
                f"PostgreSQL has {actual_rows} {model_id} rows; "
                f"expected exactly {expected_rows}"
            )

        if actual_rows == expected_rows:
            if actual_events != expected_events:
                raise AssertionError(
                    f"PostgreSQL event count={actual_events}; expected={expected_events}"
                )
            actual_images = int(report["verified_unique_images"])
            if actual_images != expected_images:
                raise AssertionError(
                    f"PostgreSQL references {actual_images} unique event images; "
                    f"expected={expected_images}"
                )
            return report

        time.sleep(0.5)

    raise TimeoutError(
        f"PostgreSQL did not reach {expected_rows} {model_id} results within "
        f"{timeout_seconds:.1f}s; last_report={last_report}; "
        f"last_database_error={last_database_error}"
    )


def stop_results_processor(processor: subprocess.Popen | None) -> int | None:
    if processor is None:
        return None
    if processor.poll() is None:
        processor.send_signal(signal.SIGINT)
        try:
            processor.wait(timeout=180)
        except subprocess.TimeoutExpired:
            processor.terminate()
            try:
                processor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                processor.kill()
                processor.wait()
    return processor.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Complete one-camera DeepStream + YOLO11n TensorRT + Kafka + "
            "PostgreSQL event pipeline"
        )
    )
    parser.add_argument("--config", default=None, help="Application YAML configuration")
    parser.add_argument("--db-timeout", type=float, default=None)
    parser.add_argument("--processor-startup-timeout", type=float, default=60.0)
    parser.add_argument("--consumer-group", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--metrics-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> dict:
    parser = build_parser()
    args, producer_args = parser.parse_known_args(argv)

    if args.db_timeout is not None and args.db_timeout <= 0:
        parser.error("--db-timeout must be positive")
    if args.processor_startup_timeout <= 0:
        parser.error("--processor-startup-timeout must be positive")
    if "--no-kafka-verify" in producer_args:
        parser.error(
            "The end-to-end runner requires Kafka result verification; "
            "do not pass --no-kafka-verify."
        )

    bootstrap_override = _option_value(
        producer_args, "--kafka-bootstrap-servers", None
    )
    if bootstrap_override:
        os.environ["KAFKA_BOOTSTRAP_SERVERS"] = bootstrap_override

    config_candidate = (
        args.config
        or os.environ.get("APP_CONFIG_PATH")
        or "configs/config.yaml"
    )
    settings = load_settings(config_candidate)
    config_path = settings.paths.resolve(config_candidate)
    os.environ["APP_CONFIG_PATH"] = str(config_path)

    kafka_bootstrap = bootstrap_override or settings.kafka.bootstrap_servers
    kafka_config = load_kafka_config(config_path)
    configured_topic = topic_for_model(kafka_config, settings.yolo.model_id)
    requested_topic = _option_value(producer_args, "--topic", configured_topic)
    if requested_topic != configured_topic:
        parser.error(
            f"--topic must match kafka.topics.{settings.yolo.model_id} "
            f"({configured_topic!r}) so the results processor consumes it."
        )

    session_id = UUID(args.session_id).hex if args.session_id else uuid4().hex
    safe_model_id = re.sub(r"[^A-Za-z0-9._-]+", "-", settings.yolo.model_id)
    group_id = args.consumer_group or f"deepstream-{safe_model_id}-{session_id[:12]}"
    db_timeout = (
        args.db_timeout
        if args.db_timeout is not None
        else settings.benchmark.database_timeout_seconds
    )

    # Create all configured result topics and the DLT before the processor
    # subscribes. The producer's own topic check is intentionally idempotent.
    topics_to_ensure = list(dict.fromkeys(kafka_config["topics"].values()))
    dead_letter_topic = (
        os.environ.get("INFERENCE_DEAD_LETTER_TOPIC")
        or kafka_config.get("dead_letter", {}).get("topic")
    )
    if dead_letter_topic:
        topics_to_ensure.append(dead_letter_topic)
    topic_status = KafkaTopicManager(
        kafka_bootstrap,
        list(dict.fromkeys(topics_to_ensure)),
        kafka_config.get("kafka_admin", {}),
    ).ensure_topics()

    database_config = settings.database.connection_kwargs()
    database = DatabaseClient(**database_config)
    try:
        if not database.check_connection():
            raise RuntimeError("PostgreSQL connection check failed")
    except psycopg.OperationalError as exc:
        if "password authentication failed" in str(exc).casefold():
            raise RuntimeError(
                "PostgreSQL rejected the configured application password. "
                f"Target: {database_config['user']}@{database_config['host']}:"
                f"{database_config['port']}/{database_config['dbname']}. "
                f"Set {settings.database.password_env} in the project .env file "
                f"to the password assigned to PostgreSQL role "
                f"{database_config['user']}. Do not use POSTGRES_PASSWORD unless "
                "it is also the password for this application role."
            ) from exc
        raise
    _validate_database_schema(database)

    environment = os.environ.copy()
    environment["APP_CONFIG_PATH"] = str(config_path)
    environment["KAFKA_BOOTSTRAP_SERVERS"] = kafka_bootstrap
    environment["RESULTS_CONSUMER_GROUP_ID"] = group_id
    # The generated group is unique to this run. Starting at latest avoids
    # replaying unrelated historical result messages before this producer runs.
    environment["KAFKA_AUTO_OFFSET_RESET"] = "latest"

    processor_command = [
        sys.executable,
        "-m",
        "src.results.run_results_processor",
        "--config",
        str(config_path),
    ]

    processor: subprocess.Popen | None = None
    failure: BaseException | None = None

    try:
        print(
            f"[E2E] session={session_id} model={settings.yolo.model_id} "
            f"topic={configured_topic} consumer_group={group_id}",
            flush=True,
        )
        processor = subprocess.Popen(
            processor_command,
            env=environment,
            cwd=settings.paths.root,
        )
        group_ready = wait_for_consumer_group(
            kafka_bootstrap,
            group_id,
            args.processor_startup_timeout,
            processor,
        )

        # Keep the producer's parser/configuration as the single source of
        # truth; only inject this run's fresh session ID.
        producer_args = _without_option(producer_args, "--session-id")
        producer_args.extend(["--session-id", session_id])
        from src.pipeline.run_tensorrt_event_pipeline import main as run_producer

        producer_report = run_producer(producer_args)
        expected_rows = int(producer_report.get("frames_sampled", 0))
        expected_events = int(producer_report.get("event_frames_saved", 0))
        store_stats = producer_report.get("event_image_store_stats", {})
        expected_images = int(store_stats.get("saved", 0)) + int(
            store_stats.get("reused", 0)
        )

        if expected_rows <= 0:
            raise RuntimeError("TensorRT producer completed without sampled frames")
        if expected_images != expected_events:
            raise AssertionError(
                f"Event image count mismatch: events={expected_events}, "
                f"stored/reused images={expected_images}"
            )

        store = EventFrameStore(producer_report["event_image_store"])
        database_report = wait_for_database_session(
            database=database,
            store=store,
            session_id=session_id,
            model_id=settings.yolo.model_id,
            expected_rows=expected_rows,
            expected_events=expected_events,
            expected_images=expected_images,
            timeout_seconds=db_timeout,
            processor=processor,
        )

        metrics_root = settings.paths.resolve(
            args.metrics_dir or settings.metrics_directory
        )
        metrics_root.mkdir(parents=True, exist_ok=True)
        report_path = metrics_root / f"tensorrt_end_to_end_{session_id}.json"
        final_report = {
            "schema_version": "1.0",
            "source_session_id": session_id,
            "model_id": settings.yolo.model_id,
            "camera_id": producer_report["camera_id"],
            "kafka_topic": configured_topic,
            "kafka_topic_status": topic_status,
            "results_consumer_group": group_id,
            "results_consumer_group_ready": group_ready,
            "expected_results": expected_rows,
            "expected_event_results": expected_events,
            "expected_unique_event_images": expected_images,
            "postgresql": database_report,
            "producer": producer_report,
            "report_path": str(report_path),
        }
        report_path.write_text(
            json.dumps(final_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("\n" + "=" * 78)
        print("COMPLETE TENSORRT END-TO-END PIPELINE REPORT")
        print("=" * 78)
        print(json.dumps(final_report, indent=2, ensure_ascii=False))
        print("=" * 78)
        print(
            "PASS: DeepStream -> YOLO11n TensorRT -> event JPEG/JSON -> "
            "Kafka -> PostgreSQL"
        )
        return final_report

    except BaseException as exc:
        failure = exc
        raise

    finally:
        processor_return_code = stop_results_processor(processor)
        if failure is None and processor is not None and processor_return_code != 0:
            raise RuntimeError(
                "Results processor did not stop cleanly "
                f"(exit code={processor_return_code})"
            )
        if failure is not None and processor is not None and processor_return_code not in (0, None):
            print(
                f"[E2E WARNING] Results processor exit code={processor_return_code}",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    main()
