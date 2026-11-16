"""Smoke test for independent asynchronous model-result JSONL sinks.

This test does not require CUDA, DeepStream, or Kafka. It verifies the result
schema, independent per-model queues, bounded backpressure, batching, final
flush on shutdown, and file integrity before the sinks are connected to the
DeepStream inference workers.
"""

from __future__ import annotations

import argparse
import json
import threading
import traceback
from pathlib import Path
from typing import Any

from src.results.async_result_sink import AsyncJSONLResultSink
from src.results.model_result_message import (
    build_result_message,
    validate_result_message,
)


MODEL_IDS = {
    "model_a": "mock_model_a",
    "model_b": "mock_model_b",
}


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return number


def sample_inference_result(model_id: str, frame_id: int) -> dict[str, Any]:
    """Create a deterministic CPU-side result with the same shape as the models."""

    confidence_base = 0.70 if model_id == "mock_model_a" else 0.68
    detections = [
        {
            "class_id": 0,
            "class_name": "person",
            "confidence": round(confidence_base, 5),
            "bbox_normalized": [0.10, 0.12, 0.35, 0.72],
            "synthetic": True,
        },
        {
            "class_id": 1,
            "class_name": "vehicle",
            "confidence": 0.61,
            "bbox_normalized": [0.52, 0.24, 0.91, 0.81],
            "synthetic": True,
        },
    ]

    return {
        "model_id": model_id,
        "frame_id": frame_id,
        "gpu_id": 0,
        "input_frame_shape": [720, 1280, 3],
        "model_input_size": 320,
        "timings_ms": {
            "preprocessing_gpu_ms": 0.35,
            "inference_gpu_ms": 0.52,
            "postprocessing_gpu_ms": 0.41,
            "gpu_total_ms": 1.28,
            "output_transfer_ms": 0.06,
            "cpu_result_build_ms": 0.02,
            "end_to_end_ms": 1.39,
        },
        "detection_count": len(detections),
        "detections": detections,
    }


def producer(
    *,
    logical_model_name: str,
    model_id: str,
    sink: AsyncJSONLResultSink,
    message_count: int,
    errors: list[str],
    errors_lock: threading.Lock,
) -> None:
    try:
        for frame_id in range(message_count):
            inference_result = sample_inference_result(model_id, frame_id)
            message = build_result_message(
                inference_result,
                source_id=f"sink-test-{logical_model_name}",
                frame_shape=(720, 1280, 3),
                queue_wait_ms=0.15,
                worker_processing_ms=1.50,
            )
            sink.submit(message)
    except BaseException as exc:
        with errors_lock:
            errors.append(f"{logical_model_name}: {type(exc).__name__}: {exc}")
        traceback.print_exc()


def validate_output_file(
    path: Path,
    *,
    expected_model_id: str,
    expected_count: int,
) -> tuple[int, int]:
    frame_ids: list[int] = []
    line_count = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise RuntimeError(f"Blank JSONL record at {path}:{line_number}")
            message = json.loads(line)
            validate_result_message(message)

            if message["model_id"] != expected_model_id:
                raise RuntimeError(
                    f"Wrong model in {path}:{line_number}: "
                    f"{message['model_id']!r}"
                )
            frame_ids.append(message["frame_id"])
            line_count += 1

    expected_frame_ids = list(range(expected_count))
    if sorted(frame_ids) != expected_frame_ids:
        raise RuntimeError(
            f"Frame IDs in {path} do not match the expected sequence."
        )

    return line_count, len(set(frame_ids))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test independent asynchronous result queues and JSONL batch sinks."
    )
    parser.add_argument("--messages-per-model", type=positive_int, default=100)
    parser.add_argument("--queue-size", type=positive_int, default=16)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--flush-interval-ms", type=positive_int, default=100)
    parser.add_argument(
        "--output-dir",
        default="artifacts/result_sink_test",
        help="Directory for model_a_results.jsonl and model_b_results.jsonl.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sinks = {
        "model_a": AsyncJSONLResultSink(
            model_id=MODEL_IDS["model_a"],
            output_path=output_dir / "model_a_results.jsonl",
            queue_size=args.queue_size,
            batch_size=args.batch_size,
            flush_interval_ms=args.flush_interval_ms,
        ),
        "model_b": AsyncJSONLResultSink(
            model_id=MODEL_IDS["model_b"],
            output_path=output_dir / "model_b_results.jsonl",
            queue_size=args.queue_size,
            batch_size=args.batch_size,
            flush_interval_ms=args.flush_interval_ms,
        ),
    }

    errors: list[str] = []
    errors_lock = threading.Lock()
    started_sinks: list[AsyncJSONLResultSink] = []
    producers: list[threading.Thread] = []

    print("=" * 64)
    print("INDEPENDENT ASYNCHRONOUS RESULT SINK TEST")
    print("=" * 64)
    print(f"Messages per model: {args.messages_per_model}")
    print(f"Queue capacity per model: {args.queue_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Flush interval: {args.flush_interval_ms} ms")
    print(f"Output directory: {output_dir.resolve()}")

    try:
        for sink in sinks.values():
            sink.start()
            started_sinks.append(sink)

        for logical_model_name in ("model_a", "model_b"):
            thread = threading.Thread(
                target=producer,
                name=f"{logical_model_name}-result-producer",
                kwargs={
                    "logical_model_name": logical_model_name,
                    "model_id": MODEL_IDS[logical_model_name],
                    "sink": sinks[logical_model_name],
                    "message_count": args.messages_per_model,
                    "errors": errors,
                    "errors_lock": errors_lock,
                },
                daemon=False,
            )
            thread.start()
            producers.append(thread)

        for thread in producers:
            thread.join()

    finally:
        for sink in started_sinks:
            sink.close()

    if errors:
        raise RuntimeError("Result producer errors: " + "; ".join(errors))

    print("\n========== SINK RESULTS ==========")
    successful = True

    for logical_model_name, sink in sinks.items():
        stats = sink.stats
        output_path = Path(stats["output_path"])
        line_count, unique_frame_count = validate_output_file(
            output_path,
            expected_model_id=MODEL_IDS[logical_model_name],
            expected_count=args.messages_per_model,
        )

        model_ok = (
            stats["enqueued_messages"] == args.messages_per_model
            and stats["written_messages"] == args.messages_per_model
            and line_count == args.messages_per_model
            and unique_frame_count == args.messages_per_model
            and stats["error"] is None
        )
        successful = successful and model_ok

        print(f"\n{logical_model_name.upper()}")
        print(f"Model ID: {MODEL_IDS[logical_model_name]}")
        print(f"Enqueued: {stats['enqueued_messages']}")
        print(f"Written: {stats['written_messages']}")
        print(f"JSONL records: {line_count}")
        print(f"Unique frame IDs: {unique_frame_count}")
        print(f"Batches written: {stats['batches_written']}")
        print(f"Maximum batch size: {stats['max_batch_size']}")
        print(f"Maximum queue depth: {stats['max_queue_depth']}")
        print(f"Blocked submissions: {stats['blocked_submissions']}")
        print(
            "Average submit wait: "
            f"{(stats['submit_wait_total_ms'] / stats['enqueued_messages']) if stats['enqueued_messages'] else 0.0:.3f} ms"
        )
        print(f"Output: {output_path}")
        print(f"Sink error: {stats['error']}")
        print(f"Validation: {'PASSED' if model_ok else 'FAILED'}")

    print("\n" + "=" * 64)
    if successful:
        print("INDEPENDENT ASYNCHRONOUS RESULT SINK TEST PASSED")
    else:
        print("INDEPENDENT ASYNCHRONOUS RESULT SINK TEST FAILED")
    print("=" * 64)

    if not successful:
        raise RuntimeError("One or more result sinks failed validation.")


if __name__ == "__main__":
    main()
