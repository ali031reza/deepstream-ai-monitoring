
"""End-to-end test for independent binary Protobuf Kafka publishers."""

from __future__ import annotations

import argparse
import time
import uuid

from collections import defaultdict

from confluent_kafka import (
    Consumer,
    KafkaError,
    KafkaException,
)

from confluent_kafka.admin import (
    AdminClient,
    NewTopic,
)

from src.proto.inference_result_pb2 import InferenceResult

from src.results.protobuf_result_adapter import (
    build_inference_result,
)

from src.results.async_protobuf_kafka_publisher import (
    AsyncProtobufKafkaPublisher,
)

from src.results.kafka_config import (
    load_kafka_config,
    producer_config_for_model,
    publisher_options,
)


MODEL_IDS = ("mock_model_a", "mock_model_b")

CAMERA_ID = "protobuf-kafka-test-camera"
SESSION_ID = "protobuf-kafka-test-session"

CAPTURED_AT_MS = 1794744599900


def create_topics(admin, topics):

    futures = admin.create_topics([
        NewTopic(
            topic,
            num_partitions=1,
            replication_factor=1,
        )
        for topic in topics
    ])

    for topic, future in futures.items():
        future.result(timeout=35)
        print(f"[ADMIN] Created topic: {topic}")


def delete_topics(admin, topics):

    futures = admin.delete_topics(
        topics,
        operation_timeout=30,
    )

    for topic, future in futures.items():

        try:
            future.result(timeout=35)
            print(f"[ADMIN] Deleted topic: {topic}")

        except Exception as exc:
            print(f"[WARNING] Could not delete {topic}: {exc}")


def make_test_message(model_id, frame_sequence):

    model_output = {
        "model_id": model_id,
        "frame_id": frame_sequence,
        "gpu_id": 0,
        "input_frame_shape": [720, 1280, 3],
        "model_input_size": 320,

        "detections": [
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.95,
                "bbox_normalized": [
                    0.10,
                    0.20,
                    0.30,
                    0.40,
                ],
                "synthetic": True,
            }
        ],

        "timings_ms": {
            "preprocessing_gpu_ms": 0.45,
            "inference_gpu_ms": 0.75,
            "postprocessing_gpu_ms": 0.25,
            "end_to_end_ms": 1.50,
        },
    }

    return build_inference_result(
        model_output,
        camera_id=CAMERA_ID,
        source_session_id=SESSION_ID,
        frame_sequence=frame_sequence,
        captured_at_ms=CAPTURED_AT_MS,
        worker_id=f"{model_id}-test-worker",
    )


def run_test(
    *,
    config_path,
    messages_per_model,
    consume_timeout_seconds,
    keep_test_topics,
):

    config = load_kafka_config(config_path)

    bootstrap = config["bootstrap_servers"]

    suffix = uuid.uuid4().hex[:10]

    test_topics = {
        "mock_model_a": f"test-protobuf-model-a-{suffix}",
        "mock_model_b": f"test-protobuf-model-b-{suffix}",
    }

    topic_to_model = {
        topic: model_id
        for model_id, topic in test_topics.items()
    }

    admin = AdminClient({
        "bootstrap.servers": bootstrap
    })

    consumer = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": f"protobuf-test-{suffix}",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })

    publishers = {}
    publisher_stats = {}

    try:

        create_topics(admin, list(test_topics.values()))

        consumer.subscribe(list(test_topics.values()))

        options = publisher_options(config)

        print("\n========== INITIALIZING PUBLISHERS ==========")

        for model_id in MODEL_IDS:

            publisher = AsyncProtobufKafkaPublisher(
                model_id=model_id,
                topic=test_topics[model_id],
                producer_config=producer_config_for_model(
                    config,
                    model_id,
                ),
                **options,
            )

            publisher.start()
            publishers[model_id] = publisher

            print(
                f"{model_id}: "
                f"topic={test_topics[model_id]}"
            )

        print("\n========== PUBLISHING BINARY PROTOBUF ==========")

        for frame_sequence in range(messages_per_model):

            for model_id in MODEL_IDS:

                message = make_test_message(
                    model_id,
                    frame_sequence,
                )

                publishers[model_id].publish(message)

            if (
                (frame_sequence + 1) % 25 == 0
                or frame_sequence + 1 == messages_per_model
            ):
                print(
                    f"Enqueued {frame_sequence + 1}/"
                    f"{messages_per_model} frames per model"
                )

        print("\n========== FLUSHING PUBLISHERS ==========")

        for model_id in MODEL_IDS:

            publisher_stats[model_id] = (
                publishers[model_id].close()
            )

            stats = publisher_stats[model_id]

            print(
                f"{model_id}: "
                f"enqueued={stats['enqueued']}, "
                f"delivered={stats['delivered']}, "
                f"delivery_failures={stats['delivery_failures']}, "
                f"publish_failures={stats['publish_failures']}, "
                f"batches={stats['batches_submitted']}"
            )

        print("\n========== CONSUMING PROTOBUF MESSAGES ==========")

        received = defaultdict(list)

        deadline = (
            time.monotonic()
            + consume_timeout_seconds
        )

        expected_total = messages_per_model * 2

        while sum(
            len(items) for items in received.values()
        ) < expected_total:

            remaining = deadline - time.monotonic()

            if remaining <= 0:
                break

            kafka_message = consumer.poll(
                min(1.0, remaining)
            )

            if kafka_message is None:
                continue

            if kafka_message.error():

                if (
                    kafka_message.error().code()
                    == KafkaError._PARTITION_EOF
                ):
                    continue

                raise KafkaException(kafka_message.error())

            topic = kafka_message.topic()

            if topic not in topic_to_model:
                raise AssertionError(
                    f"Unexpected topic: {topic}"
                )

            # Decode actual binary Protobuf bytes.
            decoded = InferenceResult()

            decoded.ParseFromString(
                kafka_message.value()
            )

            expected_model = topic_to_model[topic]

            if decoded.model_id != expected_model:
                raise AssertionError(
                    "Topic/model identity mismatch"
                )

            if decoded.camera_id != CAMERA_ID:
                raise AssertionError("Camera ID mismatch")

            if decoded.source_session_id != SESSION_ID:
                raise AssertionError("Session ID mismatch")

            if kafka_message.key() != CAMERA_ID.encode("utf-8"):
                raise AssertionError("Kafka partition key mismatch")

            if decoded.schema_version != "1.0":
                raise AssertionError("Schema version mismatch")

            if decoded.status != "processed":
                raise AssertionError("Unexpected inference status")

            if len(decoded.detections) != 1:
                raise AssertionError("Detection count mismatch")

            detection = decoded.detections[0]

            if detection.class_name != "person":
                raise AssertionError("Detection class mismatch")

            if not detection.synthetic:
                raise AssertionError("Synthetic flag was not preserved")

            if abs(detection.confidence - 0.95) > 0.0001:
                raise AssertionError("Confidence mismatch")

            if len(decoded.timings) != 4:
                raise AssertionError("Timing metadata mismatch")

            received[expected_model].append(decoded)

        total_received = sum(
            len(items) for items in received.values()
        )

        if total_received != expected_total:
            raise AssertionError(
                f"Expected {expected_total} messages, "
                f"received {total_received}"
            )

        all_result_ids = set()

        indexed_results = {}

        print("\n========== VALIDATING RESULTS ==========")

        for model_id in MODEL_IDS:

            messages = received[model_id]

            if len(messages) != messages_per_model:
                raise AssertionError(
                    f"{model_id}: incorrect message count"
                )

            sequences = [
                message.frame_sequence
                for message in messages
            ]

            if set(sequences) != set(range(messages_per_model)):
                raise AssertionError(
                    f"{model_id}: missing or invalid frame sequences"
                )

            if len(sequences) != len(set(sequences)):
                raise AssertionError(
                    f"{model_id}: duplicate frames"
                )

            indexed_results[model_id] = {
                message.frame_sequence: message
                for message in messages
            }

            for message in messages:

                if message.result_id in all_result_ids:
                    raise AssertionError("Duplicate result ID")

                all_result_ids.add(message.result_id)

            stats = publisher_stats[model_id]

            if stats["enqueued"] != messages_per_model:
                raise AssertionError("Enqueue count mismatch")

            if stats["delivered"] != messages_per_model:
                raise AssertionError("Delivery callback count mismatch")

            if stats["delivery_failures"] != 0:
                raise AssertionError("Kafka delivery failures")

            if stats["publish_failures"] != 0:
                raise AssertionError("Kafka publish failures")

            if stats["worker_errors"] != 0:
                raise AssertionError("Publisher worker errors")

            if stats["shutdown_pending"] != 0:
                raise AssertionError("Undelivered messages at shutdown")

            if stats["outstanding_after_produce"] != 0:
                raise AssertionError("Outstanding Kafka messages")

            print(
                f"{model_id}: "
                f"consumed={len(messages)}, "
                f"unique_frames={len(set(sequences))}, "
                f"topic={test_topics[model_id]}"
            )

        # Verify that both models refer to the same logical frames.
        for frame_sequence in range(messages_per_model):

            a = indexed_results["mock_model_a"][frame_sequence]
            b = indexed_results["mock_model_b"][frame_sequence]

            if a.frame_id != b.frame_id:
                raise AssertionError("Shared frame identity mismatch")

            if a.camera_id != b.camera_id:
                raise AssertionError("Shared camera identity mismatch")

            if a.result_id == b.result_id:
                raise AssertionError("Model result IDs must be independent")

        print("\n" + "=" * 68)
        print("BINARY PROTOBUF KAFKA TEST PASSED")
        print("=" * 68)
        print(f"Messages per model: {messages_per_model}")
        print(f"Total verified messages: {total_received}")
        print("Binary Protobuf serialization: VERIFIED")
        print("Protobuf deserialization: VERIFIED")
        print("Independent Kafka topics: VERIFIED")
        print("Model identity: VERIFIED")
        print("Shared frame identity: VERIFIED")
        print("Independent result IDs: VERIFIED")
        print("Detection metadata: VERIFIED")
        print("Synthetic flags: VERIFIED")
        print("GPU timing metadata: VERIFIED")
        print("Kafka delivery callbacks: VERIFIED")
        print("=" * 68)

    finally:

        for publisher in publishers.values():

            try:
                publisher.close()

            except Exception as exc:
                print(
                    f"[WARNING] Publisher shutdown: {exc}"
                )

        consumer.close()

        if not keep_test_topics:
            delete_topics(
                admin,
                list(test_topics.values()),
            )
        else:
            print(
                "[INFO] Test topics retained:",
                list(test_topics.values()),
            )


def main():

    parser = argparse.ArgumentParser(
        description="Test binary Protobuf Kafka publishing"
    )

    parser.add_argument(
        "--config",
        default=None,
    )

    parser.add_argument(
        "--messages-per-model",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--consume-timeout",
        type=float,
        default=60.0,
    )

    parser.add_argument(
        "--keep-test-topics",
        action="store_true",
    )

    args = parser.parse_args()

    if args.messages_per_model <= 0:
        parser.error("--messages-per-model must be positive")

    if args.consume_timeout <= 0:
        parser.error("--consume-timeout must be positive")

    run_test(
        config_path=args.config,
        messages_per_model=args.messages_per_model,
        consume_timeout_seconds=args.consume_timeout,
        keep_test_topics=args.keep_test_topics,
    )


if __name__ == "__main__":
    main()
