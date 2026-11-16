import base64
import json
import unittest

from src.kafka.dead_letter_publisher import DeadLetterPublisher
from src.kafka.records import KafkaRecord


class FakeProducer:
    def __init__(self, config):
        self.config = config
        self.messages = []
        self.callbacks = []
        self.fail_delivery = False
        self.pending = 0

    def produce(self, topic, **kwargs):
        self.messages.append((topic, kwargs))
        self.callbacks.append(kwargs["on_delivery"])
        self.pending += 1

    def poll(self, seconds):
        return 0

    def flush(self, timeout):
        for callback in self.callbacks:
            callback("broker unavailable" if self.fail_delivery else None, None)
        self.callbacks.clear()
        self.pending = 0
        return self.pending


class DeadLetterPublisherTest(unittest.TestCase):
    def setUp(self):
        self.producer = None
        def factory(config):
            self.producer = FakeProducer(config)
            return self.producer
        self.publisher = DeadLetterPublisher(
            bootstrap_servers="localhost:9092",
            topic="inference-dead-letter",
            producer_factory=factory,
        )
        self.record = KafkaRecord(
            topic="model-a-results", partition=2, offset=71,
            payload=b"\x00\xffinvalid", key=b"camera-1",
            headers=(("trace-id", b"abc"),), timestamp_ms=1234,
            error_type="ValueError", error_message="invalid Protobuf",
        )

    def tearDown(self):
        self.publisher.close()

    def test_preserves_original_bytes_and_source_coordinates(self):
        self.publisher.publish_batch([self.record])
        topic, message = self.producer.messages[0]
        self.assertEqual(topic, "inference-dead-letter")
        envelope = json.loads(message["value"])
        self.assertEqual(envelope["event_id"], "model-a-results:2:71")
        self.assertEqual(envelope["source"]["offset"], 71)
        self.assertEqual(
            base64.b64decode(envelope["original_message"]["payload_base64"]),
            b"\x00\xffinvalid",
        )
        self.assertEqual(envelope["error"]["type"], "ValueError")
        self.assertEqual(
            base64.b64decode(
                envelope["original_message"]["headers"][0]["value_base64"]
            ), b"abc",
        )

    def test_delivery_failure_is_not_acknowledged(self):
        self.producer.fail_delivery = True
        with self.assertRaisesRegex(RuntimeError, "Dead-letter delivery failed"):
            self.publisher.publish_batch([self.record])


if __name__ == "__main__":
    unittest.main(verbosity=2)
