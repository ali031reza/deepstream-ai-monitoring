"""No Kafka broker or PostgreSQL server required."""
from __future__ import annotations

import logging
import threading
import time
import unittest

from src.kafka.async_batch_writer import AsyncBatchWriter
from src.kafka.consumer import KafkaConsumerError
from src.proto.inference_result_pb2 import InferenceResult
from src.results.results_processing_service import (
    BatchProcessingError,
    ResultsProcessingService,
)


class Message:
    def __init__(self, offset, payload):
        self._offset, self._payload = offset, payload

    def error(self):
        return None

    def topic(self):
        return "model-a-results"

    def partition(self):
        return 0

    def offset(self):
        return self._offset

    def value(self):
        return self._payload

    def key(self):
        return b"phase1-test-camera"

    def headers(self):
        return []

    def timestamp(self):
        return 0, 1000


class Consumer:
    def __init__(self, messages, *, commit_fails=False, on_events=None):
        self.messages = list(messages)
        self.commit_fails = commit_fails
        self.on_events = on_events
        self.committed = []
        self.events_polled = 0
        self.rebalance_epoch = 0
        self.paused = False
        self.closed = False

    def subscribe(self):
        pass

    def poll(self, timeout):
        return self.messages.pop(0) if self.messages else None

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def service_events(self, timeout=0.05):
        self.events_polled += 1
        if self.on_events:
            self.on_events(self)
        # Represent a paused broker poll, never delivering unexpected data.
        time.sleep(min(timeout, 0.001))

    def commit_batch(self, records):
        if self.commit_fails:
            raise KafkaConsumerError("Simulated offset-commit failure")
        if any(r.assignment_epoch != self.rebalance_epoch for r in records):
            raise KafkaConsumerError("Stale generation")
        self.committed.extend(records)

    def close(self):
        self.closed = True


class DLT:
    def __init__(self, fails=False):
        self.fails = fails
        self.records = []
        self.closed = False

    def publish_batch(self, records):
        if self.fails:
            raise RuntimeError("Simulated DLT broker failure")
        self.records.extend(records)

    def close(self):
        self.closed = True


def valid_message(offset):
    result = InferenceResult(
        camera_id="camera-01", source_session_id="test-session",
        frame_id=f"frame-{offset}", model_id="mock_model_a",
        worker_id="worker-01", result_id=f"test-result-{offset}",
        status="processed", processed_at_ms=1800000000000,
        captured_at_ms=1799999999900,
        frame_width=640, frame_height=480, schema_version="1.0",
    )
    return Message(offset, result.SerializeToString())


class ResultsProcessingServiceTest(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def run_service(
        self, messages, *, persist=None, dlt_fails=False,
        commit_fails=False, on_events=None, queue_capacity=3,
    ):
        consumer = Consumer(
            messages, commit_fails=commit_fails, on_events=on_events
        )
        dlt = DLT(fails=dlt_fails)
        persisted = []
        writer = AsyncBatchWriter(
            persist_batch=persist or (lambda results: persisted.extend(results)),
            dead_letter_publisher=dlt,
            batch_size=1, flush_interval_ms=100,
            queue_capacity=queue_capacity,
            retry_attempts=0, retry_backoff_ms=10,
        )
        service = ResultsProcessingService(
            consumer=consumer, batch_writer=writer,
            dead_letter_publisher=dlt,
            topic_model_map={"model-a-results": "mock_model_a"},
            poll_timeout_seconds=0.001,
            shutdown_timeout_seconds=3,
        )
        return service, consumer, dlt, persisted

    def test_valid_record_committed_after_db_success(self):
        service, consumer, dlt, persisted = self.run_service(
            [valid_message(1)]
        )
        service.run(max_messages=1)
        self.assertEqual(len(persisted), 1)
        self.assertEqual([r.offset for r in consumer.committed], [1])
        self.assertEqual(dlt.records, [])
        self.assertTrue(consumer.closed)

    def test_invalid_protobuf_is_dead_lettered_then_committed(self):
        service, consumer, dlt, persisted = self.run_service(
            [Message(2, b"\xff\x00malformed")]
        )
        service.run(max_messages=1)
        self.assertEqual(persisted, [])
        self.assertEqual(len(dlt.records), 1)
        self.assertEqual(dlt.records[0].payload, b"\xff\x00malformed")
        self.assertEqual([r.offset for r in consumer.committed], [2])

    def test_db_outage_does_not_commit(self):
        def failed_db(results):
            raise RuntimeError("Simulated database outage")
        service, consumer, dlt, _ = self.run_service(
            [valid_message(3)], persist=failed_db
        )
        with self.assertRaises(BatchProcessingError):
            service.run(max_messages=1)
        self.assertEqual(consumer.committed, [])
        self.assertEqual(dlt.records, [])
        self.assertTrue(consumer.closed)

    def test_dead_letter_outage_does_not_commit(self):
        service, consumer, _, _ = self.run_service(
            [Message(4, b"\xff\x00bad")], dlt_fails=True
        )
        with self.assertRaises(BatchProcessingError):
            service.run(max_messages=1)
        self.assertEqual(consumer.committed, [])

    def test_commit_failure_allows_db_replay(self):
        service, consumer, _, persisted = self.run_service(
            [valid_message(5)], commit_fails=True
        )
        with self.assertRaisesRegex(KafkaConsumerError, "offset-commit"):
            service.run(max_messages=1)
        self.assertEqual(len(persisted), 1)
        self.assertEqual(consumer.committed, [])

    def test_rebalance_during_db_write_blocks_stale_commit(self):
        gate = threading.Event()
        def persist(results):
            self.assertTrue(gate.wait(2), "Test timed out waiting for revocation")

        def revoke(consumer):
            if consumer.rebalance_epoch == 0:
                consumer.rebalance_epoch += 1
                gate.set()

        service, consumer, _, _ = self.run_service(
            [valid_message(6)], persist=persist, on_events=revoke
        )
        with self.assertRaisesRegex(BatchProcessingError, "assignment changed"):
            service.run(max_messages=1)
        self.assertEqual(consumer.committed, [])
        self.assertEqual(consumer.rebalance_epoch, 1)

    def test_backpressure_services_consumer_events(self):
        gate = threading.Event()
        def persist(results):
            self.assertTrue(gate.wait(2), "Backpressure test stuck")

        def on_events(consumer):
            if consumer.events_polled >= 2:
                gate.set()

        service, consumer, _, _ = self.run_service(
            [valid_message(i) for i in range(7, 12)],
            persist=persist,
            on_events=on_events, queue_capacity=1,
        )
        service.run(max_messages=5)
        self.assertEqual(len(consumer.committed), 5)
        self.assertGreaterEqual(consumer.events_polled, 2)
        self.assertGreater(service._metrics["backpressure_waits"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
