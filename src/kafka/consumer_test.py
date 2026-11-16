
"""Unit tests for the Kafka result consumer."""

import unittest

from src.kafka.consumer import (
    ConsumerConfig,
    KafkaConsumerError,
    KafkaResultConsumer,
)


class FakeMessage:

    def __init__(self, error=None):
        self._error = error

    def error(self):
        return self._error

    def topic(self):
        return "model-a-results"

    def partition(self):
        return 0

    def offset(self):
        return 10


class FakeKafkaError:

    def code(self):
        return 999

    def __str__(self):
        return "Simulated Kafka error"


class FakeConsumer:

    def __init__(self, messages=None):
        self.messages = list(messages or [])
        self.subscribed_topics = None
        self.commit_calls = []
        self.events = []
        self.closed = False

    def subscribe(self, topics):
        self.subscribed_topics = topics

    def poll(self, timeout):
        if self.messages:
            return self.messages.pop(0)
        return None

    def commit(self, message, asynchronous):
        self.events.append("commit")

        self.commit_calls.append({
            "message": message,
            "asynchronous": asynchronous,
        })

        return []

    def close(self):
        self.closed = True


def build_config():
    return ConsumerConfig(
        bootstrap_servers="localhost:29092",
        group_id="deepstream-results-test",
        topics=(
            "model-a-results",
            "model-b-results",
        ),
    )


class ConsumerTest(unittest.TestCase):

    def test_config_requires_topics(self):

        with self.assertRaises(ValueError):
            ConsumerConfig(
                bootstrap_servers="localhost:29092",
                group_id="test-group",
                topics=(),
            )

    def test_duplicate_topics_are_rejected(self):

        with self.assertRaises(ValueError):
            ConsumerConfig(
                bootstrap_servers="localhost:29092",
                group_id="test-group",
                topics=("model-a-results", "model-a-results"),
            )

    def test_manual_commit_is_enabled(self):

        config = build_config().to_client_config()

        self.assertFalse(config["enable.auto.commit"])
        self.assertFalse(config["enable.auto.offset.store"])

    def test_consumer_subscribes_to_both_topics(self):

        fake = FakeConsumer()

        consumer = KafkaResultConsumer(
            build_config(),
            consumer_factory=lambda config: fake,
        )

        consumer.subscribe()

        self.assertEqual(
            fake.subscribed_topics,
            ["model-a-results", "model-b-results"],
        )

        consumer.close()

    def test_successful_processing_commits_message(self):

        message = FakeMessage()
        fake = FakeConsumer([message])

        consumer = KafkaResultConsumer(
            build_config(),
            consumer_factory=lambda config: fake,
        )

        consumer.subscribe()

        def handler(received):
            fake.events.append("handler")
            self.assertIs(received, message)

        processed = consumer.process_next(handler)

        self.assertTrue(processed)
        self.assertEqual(
            fake.events,
            ["handler", "commit"],
        )

        self.assertEqual(len(fake.commit_calls), 1)
        self.assertIs(
            fake.commit_calls[0]["message"],
            message,
        )
        self.assertFalse(
            fake.commit_calls[0]["asynchronous"]
        )

        consumer.close()

    def test_failed_handler_does_not_commit(self):

        message = FakeMessage()
        fake = FakeConsumer([message])

        consumer = KafkaResultConsumer(
            build_config(),
            consumer_factory=lambda config: fake,
        )

        consumer.subscribe()

        def failing_handler(received):
            raise RuntimeError("Simulated processing failure")

        with self.assertRaises(RuntimeError):
            consumer.process_next(failing_handler)

        self.assertEqual(len(fake.commit_calls), 0)

        consumer.close()

    def test_empty_poll_does_not_commit(self):

        fake = FakeConsumer()

        consumer = KafkaResultConsumer(
            build_config(),
            consumer_factory=lambda config: fake,
        )

        consumer.subscribe()

        processed = consumer.process_next(lambda message: None)

        self.assertFalse(processed)
        self.assertEqual(len(fake.commit_calls), 0)

        consumer.close()

    def test_kafka_poll_error_is_reported(self):

        fake = FakeConsumer([
            FakeMessage(FakeKafkaError())
        ])

        consumer = KafkaResultConsumer(
            build_config(),
            consumer_factory=lambda config: fake,
        )

        consumer.subscribe()

        with self.assertRaises(KafkaConsumerError):
            consumer.process_next(lambda message: None)

        self.assertEqual(len(fake.commit_calls), 0)

        consumer.close()

    def test_consumer_must_subscribe_before_processing(self):

        fake = FakeConsumer()

        consumer = KafkaResultConsumer(
            build_config(),
            consumer_factory=lambda config: fake,
        )

        with self.assertRaises(KafkaConsumerError):
            consumer.process_next(lambda message: None)

        consumer.close()

    def test_consumer_close_is_idempotent(self):

        fake = FakeConsumer()

        consumer = KafkaResultConsumer(
            build_config(),
            consumer_factory=lambda config: fake,
        )

        consumer.close()
        consumer.close()

        self.assertTrue(fake.closed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
