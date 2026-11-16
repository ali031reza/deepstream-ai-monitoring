"""End-to-end Kafka test: publish and consume independent model result topics."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections import defaultdict
from typing import Any

from confluent_kafka import Consumer, KafkaError, KafkaException
from confluent_kafka.admin import AdminClient, NewTopic

from .async_kafka_publisher import AsyncKafkaPublisher
from .kafka_config import (
    load_kafka_config,
    producer_config_for_model,
    publisher_options,
)
from .model_result_message import ModelResultMessage


def _create_topics(admin: AdminClient, topics: list[str]) -> None:
    futures = admin.create_topics(
        [NewTopic(topic, num_partitions=1, replication_factor=1) for topic in topics],
        request_timeout=30,
    )
    for topic, future in futures.items():
        future.result(timeout=35)
        print(f"[ADMIN] Created test topic: {topic}")


def _delete_topics(admin: AdminClient, topics: list[str]) -> None:
    futures = admin.delete_topics(topics, operation_timeout=30)
    for topic, future in futures.items():
        try:
            future.result(timeout=35)
            print(f"[ADMIN] Deleted test topic: {topic}")
        except Exception as exc:
            print(f"[WARNING] Could not delete {topic}: {exc}")


def _make_test_message(model_id: str, frame_id: int) -> ModelResultMessage:
    inference_result: dict[str, Any] = {
        "model_id": model_id,
        "frame_id": frame_id,
        "gpu_id": 0,
        "input_frame_shape": [720, 1280, 3],
        "model_input_size": 320,
        "timings_ms": {
            "preprocessing_gpu_ms": 0.25,
            "inference_gpu_ms": 0.40,
            "postprocessing_gpu_ms": 0.70,
            "end_to_end_ms": 1.50,
        },
        "detections": [
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.91,
                "bbox_normalized": [0.10, 0.15, 0.30, 0.55],
                "synthetic": True,
            }
        ],
    }
    return ModelResultMessage.from_inference_result(
        inference_result,
        source_id="kafka-publisher-test-camera",
    )


def run_test(
    *,
    config_path: str | None,
    messages_per_model: int,
    consume_timeout_seconds: float,
    keep_test_topics: bool,
) -> None:
    config = load_kafka_config(config_path)
    bootstrap_servers = config["bootstrap_servers"]
    suffix = uuid.uuid4().hex[:10]
    test_topics = {
        "mock_model_a": f"test-model-a-{suffix}",
        "mock_model_b": f"test-model-b-{suffix}",
    }
    topic_to_model = {topic: model for model, topic in test_topics.items()}

    admin = AdminClient({"bootstrap.servers": bootstrap_servers})
    consumer = Consumer({
        "bootstrap.servers": bootstrap_servers,
        "group.id": f"deepstream-kafka-test-{suffix}",
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,
    })

    publishers: dict[str, AsyncKafkaPublisher] = {}
    publisher_stats: dict[str, dict[str, Any]] = {}

    try:
        _create_topics(admin, list(test_topics.values()))
        consumer.subscribe(list(test_topics.values()))

        options = publisher_options(config)
        for model_id, topic in test_topics.items():
            publisher = AsyncKafkaPublisher(
                model_id=model_id,
                topic=topic,
                producer_config=producer_config_for_model(config, model_id),
                **options,
            )
            publisher.start()
            publishers[model_id] = publisher

        print("\n========== PUBLISHING ==========")
        for frame_id in range(messages_per_model):
            for model_id in ("mock_model_a", "mock_model_b"):
                publishers[model_id].publish(
                    _make_test_message(model_id, frame_id)
                )
            if (frame_id + 1) % 25 == 0 or frame_id + 1 == messages_per_model:
                print(
                    f"Enqueued {frame_id + 1}/{messages_per_model} "
                    "messages per model"
                )

        print("\n========== FLUSHING PRODUCERS ==========")
        for model_id, publisher in publishers.items():
            publisher_stats[model_id] = publisher.close()
            stats = publisher_stats[model_id]
            print(
                f"{model_id}: enqueued={stats['enqueued']}, "
                f"delivered={stats['delivered']}, "
                f"delivery_failures={stats['delivery_failures']}, "
                f"publish_failures={stats['publish_failures']}, "
                f"batches={stats['batches_submitted']}"
            )

        expected_total = messages_per_model * 2
        received_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        deadline = time.monotonic() + consume_timeout_seconds

        print("\n========== CONSUMING AND VALIDATING ==========")
        while sum(len(items) for items in received_by_model.values()) < expected_total:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            kafka_message = consumer.poll(min(1.0, remaining))
            if kafka_message is None:
                continue
            if kafka_message.error():
                if kafka_message.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(kafka_message.error())

            topic = kafka_message.topic()
            if topic not in topic_to_model:
                raise AssertionError(f"Unexpected topic received: {topic}")

            payload = json.loads(kafka_message.value().decode("utf-8"))
            expected_model = topic_to_model[topic]
            if payload.get("model_id") != expected_model:
                raise AssertionError(
                    f"Topic/model mismatch: topic={topic}, "
                    f"payload model_id={payload.get('model_id')}"
                )
            if payload.get("schema_version") != "1.0":
                raise AssertionError("Unexpected schema version")
            if payload.get("source_id") != "kafka-publisher-test-camera":
                raise AssertionError("Unexpected source_id")

            received_by_model[expected_model].append(payload)

        total_received = sum(len(items) for items in received_by_model.values())
        if total_received != expected_total:
            raise AssertionError(
                f"Expected {expected_total} messages, received {total_received}"
            )

        all_message_ids: set[str] = set()
        for model_id in ("mock_model_a", "mock_model_b"):
            messages = received_by_model[model_id]
            frame_ids = [message["frame_id"] for message in messages]
            if len(messages) != messages_per_model:
                raise AssertionError(
                    f"{model_id}: expected {messages_per_model}, "
                    f"received {len(messages)}"
                )
            if set(frame_ids) != set(range(messages_per_model)):
                raise AssertionError(f"{model_id}: frame IDs are missing or invalid")
            if len(frame_ids) != len(set(frame_ids)):
                raise AssertionError(f"{model_id}: duplicate frame IDs received")

            for message in messages:
                message_id = message["message_id"]
                if message_id in all_message_ids:
                    raise AssertionError("Duplicate message_id detected")
                all_message_ids.add(message_id)
                if message["detection_count"] != len(message["detections"]):
                    raise AssertionError("detection_count does not match detections")

            stats = publisher_stats[model_id]
            if stats["delivered"] != messages_per_model:
                raise AssertionError(
                    f"{model_id}: Kafka delivery callback confirmed "
                    f"{stats['delivered']} messages"
                )
            if stats["delivery_failures"] or stats["publish_failures"]:
                raise AssertionError(f"{model_id}: publishing failures detected")

            print(
                f"{model_id}: consumed={len(messages)}, "
                f"unique_frames={len(set(frame_ids))}, "
                f"topic={test_topics[model_id]}"
            )

        print("\n================================================")
        print("KAFKA PUBLISH/SUBSCRIBE TEST PASSED")
        print(f"Messages per model: {messages_per_model}")
        print(f"Total messages verified: {total_received}")
        print("Independent topics: VERIFIED")
        print("Model identity and frame IDs: VERIFIED")
        print("Unique message IDs: VERIFIED")
        print("Delivery callbacks: VERIFIED")
        print("================================================")

    finally:
        for publisher in publishers.values():
            try:
                publisher.close()
            except Exception as exc:
                print(f"[WARNING] Publisher shutdown reported: {exc}")
        consumer.close()
        if not keep_test_topics:
            _delete_topics(admin, list(test_topics.values()))
        else:
            print(f"[INFO] Keeping test topics: {list(test_topics.values())}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test independent asynchronous Kafka result publishers"
    )
    parser.add_argument("--config", default=None, help="Path to configs/config.yaml")
    parser.add_argument("--messages-per-model", type=int, default=100)
    parser.add_argument("--consume-timeout", type=float, default=60.0)
    parser.add_argument("--keep-test-topics", action="store_true")
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
