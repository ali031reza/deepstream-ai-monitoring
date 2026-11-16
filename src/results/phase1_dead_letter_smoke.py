"""
Real Kafka smoke test for Phase 1 dead-letter delivery and offset commit.

Run from repository root with a reachable Kafka broker and DATABASE_PASSWORD
configured. Does NOT insert any database records. Produces exactly one invalid
message to the model-A result topic and confirms its DLT envelope and the
source group's committed offset. Run with other result publishers stopped.

    python3 -m src.results.phase1_dead_letter_smoke --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import threading
import time
import uuid

from confluent_kafka import Consumer, Producer, TopicPartition

from src.kafka.async_batch_writer import AsyncBatchWriter
from src.kafka.consumer import ConsumerConfig, KafkaResultConsumer
from src.kafka.dead_letter_publisher import DeadLetterPublisher
from src.kafka.runtime_config import load_runtime_config
from src.results.results_processing_service import ResultsProcessingService


LOG = logging.getLogger(__name__)


def snapshot_source_end_offsets(metadata_client, topics):
    """Capture a per-partition starting offset BEFORE the test is published.

    An uncommitted fresh consumer group with `auto.offset.reset=latest` may
    resolve the offset only after a test message arrives, and then skip it.
    Explicit offsets supplied in the cooperative assignment callback avoid
    that race without reading older, unrelated results from the topic.
    """
    offsets = {}
    for topic in topics:
        metadata = metadata_client.list_topics(topic=topic, timeout=10)
        info = metadata.topics.get(topic)
        if info is None or info.error is not None or not info.partitions:
            raise RuntimeError(f"Source Kafka topic is unavailable: {topic}")
        for partition in sorted(info.partitions):
            key = (topic, partition)
            _low, high = metadata_client.get_watermark_offsets(
                TopicPartition(topic, partition), timeout=10, cached=False,
            )
            offsets[key] = high
    return offsets


class SnapshotStartResultConsumer(KafkaResultConsumer):
    """Test-only group consumer with deterministic initial source offsets.

    The production consumer is intentionally unchanged: the explicit starting
    offsets are appropriate for a disposable smoke-test group, NOT for the
    real results processor, which must resume from its committed offsets.
    """

    def __init__(self, config, starting_offsets, consumer_factory=None):
        self._starting_offsets = dict(starting_offsets)
        self._initialized_partitions = set()
        super().__init__(config, consumer_factory=consumer_factory)

    def _on_assign(self, kafka_consumer, partitions):
        starts = []
        for part in partitions:
            key = (part.topic, part.partition)
            if key not in self._starting_offsets:
                raise RuntimeError(f"Unexpected source partition assigned: {key}")
            if key not in self._initialized_partitions:
                starts.append(TopicPartition(
                    part.topic, part.partition, self._starting_offsets[key],
                ))
                self._initialized_partitions.add(key)
            else:
                # Later assignments must use the group's committed position;
                # never re-seek to the initial watermark after a rebalance.
                starts.append(TopicPartition(part.topic, part.partition))

        # cooperative-sticky requires incremental_assign when specifying
        # starting offsets in on_assign (Confluent Python client semantics).
        kafka_consumer.incremental_assign(starts)
        LOG.info("Smoke-test source partitions pinned: %s", starts)
        super()._on_assign(kafka_consumer, partitions)


def run(config_path=None, timeout_seconds=45):
    config = load_runtime_config(config_path)
    bootstrap = config["bootstrap_servers"]
    model_topic = next(iter(config["topic_model_map"]))
    dlt_topic = config["dead_letter_topic"]
    test_token = uuid.uuid4().hex
    group_id = f"deepstream-phase1-dlt-{test_token}"
    test_payload = b"\xff\xfephase1-invalid-protobuf-" + test_token.encode()
    test_key = f"phase1-{test_token}".encode()

    # Explicitly assign the DLT observer from the pre-test high watermark:
    # old DLT messages can never be mistaken for this test's envelope.
    observer = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": f"phase1-observer-{test_token}",
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
    })
    producer = Producer({
        "bootstrap.servers": bootstrap,
        "acks": "all",
        "enable.idempotence": True,
        "message.timeout.ms": 15000,
    })
    persisted = []

    errors = []
    stop_event = threading.Event()

    def consume_once():
        try:
            service.run(stop_event=stop_event, max_messages=1)
        except Exception as exc:
            errors.append(exc)

    consumer_thread = None
    verifier = None
    try:
        metadata = observer.list_topics(topic=dlt_topic, timeout=10)
        info = metadata.topics.get(dlt_topic)
        if info is None or info.error is not None:
            raise RuntimeError(f"DLT topic is unavailable: {dlt_topic}")
        starts = []
        for partition in sorted(info.partitions):
            low, high = observer.get_watermark_offsets(
                TopicPartition(dlt_topic, partition), timeout=10,
                cached=False,
            )
            starts.append(TopicPartition(dlt_topic, partition, high))
        observer.assign(starts)

        # Pin each source partition to its pre-test end offset. Waiting merely
        # for the group assignment is not sufficient with `latest`: offset
        # initialization can race against the new producer message.
        source_starts = snapshot_source_end_offsets(
            observer, config["topic_model_map"],
        )
        LOG.info("Captured pre-test source high-watermarks: %s", source_starts)

        consumer = SnapshotStartResultConsumer(
            ConsumerConfig(
                bootstrap_servers=bootstrap,
                group_id=group_id,
                topics=tuple(config["topic_model_map"]),
                auto_offset_reset="earliest",  # only a fallback; explicit starts win
                poll_timeout_seconds=0.1,
            ),
            starting_offsets=source_starts,
        )
        dlt = DeadLetterPublisher(
            bootstrap_servers=bootstrap,
            topic=dlt_topic,
            producer_config=config["dead_letter_producer_config"],
            flush_timeout_seconds=config["dead_letter_flush_timeout_seconds"],
        )

        def reject_database_write(results):
            persisted.extend(results)
            raise AssertionError(
                "Test consumed an unrelated valid message; stop other publishers "
                "and retry with a new test group"
            )

        writer = AsyncBatchWriter(
            persist_batch=reject_database_write,
            dead_letter_publisher=dlt,
            batch_size=1,
            flush_interval_ms=50,
            queue_capacity=10,
            retry_attempts=1,
            retry_backoff_ms=100,
        )
        service = ResultsProcessingService(
            consumer=consumer,
            batch_writer=writer,
            dead_letter_publisher=dlt,
            topic_model_map=config["topic_model_map"],
            poll_timeout_seconds=0.1,
            shutdown_timeout_seconds=20,
        )

        consumer_thread = threading.Thread(
            target=consume_once, name="phase1-dlt-smoke-consumer"
        )
        consumer_thread.start()
        if not consumer.assignment_ready.wait(timeout=timeout_seconds / 2):
            raise TimeoutError("Test consumer did not receive Kafka assignment")

        delivery = []
        def delivered(error, message):
            if error is not None:
                delivery.append(error)
            else:
                delivery.append(message)

        producer.produce(
            model_topic,
            key=test_key,
            value=test_payload,
            headers=[("content-type", b"application/x-protobuf")],
            on_delivery=delivered,
        )
        if producer.flush(timeout_seconds / 2):
            raise TimeoutError("Test input message not fully delivered")
        if len(delivery) != 1 or not hasattr(delivery[0], "partition"):
            raise RuntimeError(f"Kafka test input delivery failed: {delivery}")
        original = delivery[0]
        event_id = (
            f"{original.topic()}:{original.partition()}:{original.offset()}"
        )
        print("Input message delivered:", event_id)

        consumer_thread.join(timeout_seconds)
        if consumer_thread.is_alive():
            raise TimeoutError(
                "Results processor did not finish the DLT test; "
                f"source={event_id}, submitted={service._metrics['submitted']}, "
                f"assigned={consumer.assignment_ready.is_set()}"
            )
        if errors:
            raise RuntimeError(f"Results processor failed: {errors[0]}") from errors[0]
        if persisted:
            raise AssertionError("Unexpected database write")
        if service._metrics["dead_lettered"] != 1:
            raise AssertionError(
                f"Expected one DLT acknowledgement: {service._metrics}"
            )

        deadline = time.monotonic() + timeout_seconds
        found = []
        while time.monotonic() < deadline:
            message = observer.poll(0.25)
            if message is None:
                continue
            if message.error() is not None:
                raise RuntimeError(f"DLT observer failed: {message.error()}")
            envelope = json.loads(message.value())
            if envelope.get("event_id") == event_id:
                found.append(envelope)
                break
        if len(found) != 1:
            raise AssertionError(f"DLT envelope not found for {event_id}")
        envelope = found[0]
        original_payload = base64.b64decode(
            envelope["original_message"]["payload_base64"]
        )
        if original_payload != test_payload:
            raise AssertionError("DLT envelope does not preserve original bytes")

        verifier = Consumer({
            "bootstrap.servers": bootstrap,
            "group.id": group_id,
            "enable.auto.commit": False,
        })
        committed = verifier.committed(
            [TopicPartition(original.topic(), original.partition())],
            timeout=10,
        )[0].offset
        if committed != original.offset() + 1:
            raise AssertionError(
                f"Unexpected source offset commit {committed}; "
                f"expected {original.offset() + 1}"
            )

        print("DLT envelope verified:", event_id)
        print("DLT payload preservation: OK")
        print("PostgreSQL writes: 0")
        print("Kafka committed next offset:", committed)
        print("PASS: real Kafka dead-letter + offset integration smoke test")
    finally:
        stop_event.set()
        if consumer_thread is not None and consumer_thread.is_alive():
            consumer_thread.join(timeout=25)
        observer.close()
        if verifier is not None:
            verifier.close()
        remaining = producer.flush(5)
        if remaining:
            LOG.warning("%d test producer messages still pending", remaining)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--timeout", type=float, default=45)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run(args.config, args.timeout)


if __name__ == "__main__":
    main()
