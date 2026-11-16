import unittest

from src.kafka.async_batch_writer import AsyncBatchWriter
from src.kafka.records import KafkaRecord
from src.proto.inference_result_pb2 import InferenceResult


class FakeDeadLetterPublisher:
    def __init__(self):
        self.batches = []

    def publish_batch(self, records):
        self.batches.append(list(records))


def make_record(offset, invalid=False):
    result = None
    if not invalid:
        result = InferenceResult(
            camera_id="camera-01",
            source_session_id="session-01",
            frame_id=f"frame-{offset}",
            model_id="mock_model_a",
            worker_id="worker-01",
            result_id=f"result-{offset}",
            status="processed",
            processed_at_ms=1794744600000,
            captured_at_ms=1794744599000,
            frame_width=1920,
            frame_height=1080,
            schema_version="1.0",
        )

    return KafkaRecord(
        topic="model-a-results",
        partition=0,
        offset=offset,
        payload=b"test-payload",
        key=b"camera-01",
        headers=(),
        timestamp_ms=None,
        result=result,
        error_type="ValidationError" if invalid else None,
        error_message="Invalid message" if invalid else None,
    )


class AsyncBatchWriterTest(unittest.TestCase):
    def setUp(self):
        self.persisted_batches = []
        self.dlt = FakeDeadLetterPublisher()

    def create_writer(
        self,
        batch_size=2,
        flush_interval_ms=100,
        retry_attempts=0,
        persist=None,
    ):
        return AsyncBatchWriter(
            persist_batch=persist or self.persisted_batches.append,
            dead_letter_publisher=self.dlt,
            batch_size=batch_size,
            flush_interval_ms=flush_interval_ms,
            queue_capacity=10,
            retry_attempts=retry_attempts,
            retry_backoff_ms=10,
        )

    def test_flushes_when_batch_size_is_reached(self):
        writer = self.create_writer(
            batch_size=2, flush_interval_ms=5000
        )
        writer.start()
        self.assertTrue(writer.submit(make_record(1)))
        self.assertTrue(writer.submit(make_record(2)))
        completion = writer.get_completion(timeout=2)
        self.assertTrue(completion.success)
        self.assertEqual(completion.persisted_count, 2)
        self.assertEqual(len(self.persisted_batches), 1)
        writer.request_stop()
        self.assertTrue(writer.join(2))

    def test_flushes_when_interval_expires(self):
        writer = self.create_writer(
            batch_size=10, flush_interval_ms=50
        )
        writer.start()
        writer.submit(make_record(1))
        completion = writer.get_completion(timeout=2)
        self.assertTrue(completion.success)
        self.assertEqual(completion.persisted_count, 1)
        writer.request_stop()
        self.assertTrue(writer.join(2))

    def test_shutdown_flushes_partial_batch(self):
        writer = self.create_writer(
            batch_size=10, flush_interval_ms=10000
        )
        writer.start()
        writer.submit(make_record(1))
        writer.request_stop()
        self.assertTrue(writer.join(2))
        completion = writer.get_completion(timeout=1)
        self.assertTrue(completion.success)
        self.assertEqual(completion.persisted_count, 1)

    def test_invalid_messages_go_to_dead_letter(self):
        writer = self.create_writer(batch_size=1)
        writer.start()
        writer.submit(make_record(1, invalid=True))
        completion = writer.get_completion(timeout=2)
        self.assertTrue(completion.success)
        self.assertEqual(completion.dead_letter_count, 1)
        self.assertEqual(len(self.persisted_batches), 0)
        self.assertEqual(len(self.dlt.batches), 1)
        writer.request_stop()
        self.assertTrue(writer.join(2))

    def test_shutdown_does_not_wait_entire_flush_interval(self):
        import time
        writer = self.create_writer(
            batch_size=100, flush_interval_ms=60000
        )
        writer.start()
        writer.submit(make_record(1))
        started = time.monotonic()
        writer.request_stop()
        self.assertTrue(writer.join(1))
        self.assertLess(time.monotonic() - started, 1)
        self.assertEqual(writer.get_completion(timeout=1).persisted_count, 1)

    def test_does_not_flush_before_long_interval(self):
        import queue
        writer = self.create_writer(
            batch_size=100, flush_interval_ms=1000
        )
        writer.start()
        writer.submit(make_record(1))
        with self.assertRaises(queue.Empty):
            writer.get_completion(timeout=0.25)
        writer.request_stop()
        self.assertTrue(writer.join(2))
        self.assertTrue(writer.get_completion(timeout=1).success)

    def test_dead_letter_failure_blocks_acknowledgement(self):
        class FailedDLT:
            def publish_batch(self, records):
                raise RuntimeError("DLT producer unavailable")
        writer = self.create_writer(batch_size=1)
        writer.dead_letter_publisher = FailedDLT()
        writer.start()
        writer.submit(make_record(1, invalid=True))
        completion = writer.get_completion(timeout=2)
        self.assertFalse(completion.success)
        self.assertEqual(completion.persisted_count, 0)
        self.assertTrue(writer.join(2))

    def test_database_failure_is_reported(self):
        def failing_persist(results):
            raise RuntimeError("Database unavailable")

        writer = self.create_writer(
            batch_size=1, persist=failing_persist
        )
        writer.start()
        writer.submit(make_record(1))
        completion = writer.get_completion(timeout=2)
        self.assertFalse(completion.success)
        self.assertIsNotNone(writer.fatal_error)
        self.assertEqual(len(self.dlt.batches), 0)
        self.assertTrue(writer.join(2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
