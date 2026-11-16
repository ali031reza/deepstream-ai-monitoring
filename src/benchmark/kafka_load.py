"""Phase 4: publish N matching two-model frames through Kafka and audit DB.

Use with the Phase-1 results processor already running. No event images are
fabricated: the synthetic load publishes is_event=False on purpose.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
from uuid import uuid4, uuid5, NAMESPACE_URL

from src.monitoring.metrics import RunMetrics, save_json
from src.monitoring.resource_sampler import ResourceSampler


def make_result(*, camera_id, session_id, frame_number, model_id, timestamp_ms):
    from src.proto.inference_result_pb2 import InferenceResult
    result = InferenceResult(
        camera_id=camera_id, source_session_id=session_id,
        frame_id=f"{session_id}:{frame_number:012d}",
        model_id=model_id,
        result_id=str(uuid5(NAMESPACE_URL, f"{camera_id}:{session_id}:{frame_number}:{model_id}")),
        worker_id="phase4-load-generator",
        status="processed", processed_at_ms=timestamp_ms,
        captured_at_ms=timestamp_ms - 10,
        frame_sequence=frame_number, gpu_id=0,
        frame_width=1280, frame_height=720, model_input_size=320,
        schema_version="1.0", is_event=False,
    )
    det = result.detections.add()
    det.class_id = 0
    det.class_name = "person"
    det.confidence = .9
    det.bbox_normalized.extend([.1, .1, .8, .8])
    det.synthetic = True
    result.timings.add(stage="inference", duration_ms=4)
    return result


def check_database(database, session_id, frame_count, timeout_seconds=120):
    deadline = time.monotonic() + timeout_seconds
    last = {}
    while time.monotonic() < deadline:
        with database.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT model_id, COUNT(*), COUNT(DISTINCT frame_id)
                    FROM inference_results WHERE source_session_id = %s
                    GROUP BY model_id
                """, (session_id,))
                rows = cur.fetchall()
        last = {model: {"records": count, "distinct_frames": distinct}
                for model, count, distinct in rows}
        if all(last.get(model, {}).get("records") == frame_count and
               last[model]["distinct_frames"] == frame_count
               for model in ("mock_model_a", "mock_model_b")):
            return last
        if any(x["records"] > frame_count for x in last.values()):
            raise AssertionError(f"Unexpected extra DB records: {last}")
        time.sleep(.5)
    raise TimeoutError(f"DB still lacks expected {frame_count} rows per model; last={last}")


def run(args):
    from confluent_kafka import Producer
    from src.results.kafka_config import load_kafka_config
    config = load_kafka_config(args.kafka_config)
    topics = config["topics"]
    if set(topics) != {"mock_model_a", "mock_model_b"}:
        raise ValueError("Kafka topics must contain mock_model_a and mock_model_b")
    session_id = uuid4().hex
    camera_id = args.camera_id
    metrics = RunMetrics()
    resources = ResourceSampler(interval_seconds=2).start()
    producer_config = dict(config.get("publisher", {}).get("producer", {}))
    producer_config.update({
        "bootstrap.servers": config["bootstrap_servers"],
        "acks": "all", "enable.idempotence": True,
        "client.id": f"phase4-load-{session_id[:8]}",
    })
    producer = Producer(producer_config)
    failed = []
    start = time.monotonic()
    try:
        for frame in range(args.frames):
            # --rate is frames/sec; each frame sends 2 Kafka messages.
            if args.rate > 0:
                due = start + frame / args.rate
                wait = due - time.monotonic()
                if wait > 0:
                    time.sleep(wait)
            for model in ("mock_model_a", "mock_model_b"):
                timestamp = time.time_ns() // 1_000_000
                result = make_result(
                    camera_id=camera_id, session_id=session_id,
                    frame_number=frame, model_id=model, timestamp_ms=timestamp,
                )
                enqueued = time.monotonic()
                def delivery(err, _msg, *, model=model, enqueued=enqueued):
                    if err:
                        failed.append(str(err))
                        metrics.error(f"Kafka {model}: {err}")
                    else:
                        metrics.add(f"{model}.delivered", duration_ms=(time.monotonic()-enqueued)*1000)
                while True:
                    try:
                        producer.produce(
                            topics[model], key=camera_id.encode(),
                            value=result.SerializeToString(deterministic=True),
                            on_delivery=delivery,
                        )
                        break
                    except BufferError:
                        metrics.add("producer.backpressure")
                        producer.poll(.05)
                producer.poll(0)
        remaining = producer.flush(args.kafka_timeout)
        if remaining or failed:
            raise RuntimeError(f"Kafka delivery failed: remaining={remaining}, errors={failed[:5]}")
        delivery_seconds = time.monotonic() - start
        database_report = None
        if args.await_database:
            from src.kafka.runtime_config import load_runtime_config
            from src.database.client import DatabaseClient
            database = DatabaseClient(**load_runtime_config(args.kafka_config)["database"])
            database_report = check_database(database, session_id, args.frames, args.db_timeout)
        elapsed = time.monotonic() - start
        report = {
            "schema": "phase4-load-v1", "session_id": session_id, "camera_id": camera_id,
            "frames": args.frames, "messages": args.frames * 2,
            "target_frame_rate": args.rate,
            "kafka_delivery_seconds": round(delivery_seconds, 3),
            "kafka_message_throughput_s": round((args.frames * 2) / max(.001, delivery_seconds), 2),
            "total_seconds_including_db": round(elapsed, 3),
            "db_rows": database_report,
            "metrics": metrics.snapshot(),
        }
        print(json.dumps({k: v for k, v in report.items() if k != "resources"}, indent=2), flush=True)
        print("PASS: Kafka load delivered" + (" and PostgreSQL counts match" if args.await_database else ""))
        return report
    finally:
        resources_snapshot = resources.stop()
        # On failure persist what was observed as well.
        output = Path(args.report_dir) / f"load_{session_id}.json"
        if "report" not in locals():
            report = {
                "session_id": session_id, "frames": args.frames,
                "target_frame_rate": args.rate, "metrics": metrics.snapshot(),
                "failed_delivery_errors": failed[:20], "incomplete": True,
            }
        report["resources"] = resources_snapshot
        save_json(output, report)
        print(f"Phase 4 report: {output.resolve()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--rate", type=float, default=100.,
                        help="Target source frames/sec; two Kafka messages per frame. 0 = unthrottled")
    parser.add_argument("--camera-id", default="phase4-load-camera")
    parser.add_argument("--kafka-config", default="config/kafka.json")
    parser.add_argument("--await-database", action="store_true")
    parser.add_argument("--db-timeout", type=float, default=180)
    parser.add_argument("--kafka-timeout", type=float, default=90)
    parser.add_argument("--report-dir", default="artifacts/benchmarks")
    args = parser.parse_args()
    if args.frames <= 0 or args.rate < 0 or args.db_timeout <= 0 or args.kafka_timeout <= 0:
        parser.error("frames must be positive, rate nonnegative, timeouts positive")
    if not args.camera_id or len(args.camera_id) > 50:
        parser.error("camera_id length must be between 1 and 50")
    run(args)


if __name__ == "__main__":
    main()
