import unittest

from src.kafka.consumer import (
    ConsumerConfig,
    KafkaResultConsumer,
)


class FakePartition:
    def __init__(self, topic, partition, offset=0, error=None):
        self.topic = topic
        self.partition = partition
        self.offset = offset
        self.error = error


class FakeMessage:
    def __init__(self, offset=10):
        self._offset = offset

    def error(self):
        return None

    def topic(self):
        return "model-a-results"

    def partition(self):
        return 0

    def offset(self):
        return self._offset

    def value(self):
        return b"payload"

    def key(self):
        return b"camera-01"

    def headers(self):
        return []

    def timestamp(self):
        return (0, 1000)


class FakeConsumer:
    def __init__(self, config):
        self.config = config
        self.subscribed = None
        self.commits = []
        self.paused = []
        self.resumed = []
        self.closed = False
        self.messages = []
        self.callbacks = {}
        self.assigned = [FakePartition("model-a-results", 0)]
        self.raise_on_commit = None

    def subscribe(self, topics, **callbacks):
        self.subscribed = topics
        self.callbacks = callbacks

    def poll(self, timeout):
        return self.messages.pop(0) if self.messages else None

    def assignment(self):
        return self.assigned

    def pause(self, partitions):
        self.paused.extend(partitions)

    def resume(self, partitions):
        self.resumed.extend(partitions)

    def commit(self, **kwargs):
        if self.raise_on_commit:
            raise self.raise_on_commit
        self.commits.append(kwargs)
        return []

    def close(self):
        self.closed = True


class ConsumerTest(unittest.TestCase):
    def setUp(self):
        self.config = ConsumerConfig(
            bootstrap_servers="localhost:9092",
            group_id="test-group",
            topics=("model-a-results", "model-b-results"),
        )
        self.fake = None

        def factory(config):
            self.fake = FakeConsumer(config)
            return self.fake

        self.consumer = KafkaResultConsumer(
            self.config,
            consumer_factory=factory,
        )
        self.consumer.subscribe()

    def tearDown(self):
        self.consumer.close()

    def test_manual_commit_is_enabled(self):
        cfg = self.config.to_client_config()
        self.assertFalse(cfg["enable.auto.commit"])
        self.assertFalse(cfg["enable.auto.offset.store"])

    def test_subscribes_to_both_topics(self):
        self.assertEqual(
            self.fake.subscribed,
            ["model-a-results", "model-b-results"],
        )

    def test_batch_commit_uses_next_offset(self):
        record = type(
            "Record",
            (),
            {
                "topic": "model-a-results",
                "partition": 0,
                "offset": 10,
            },
        )()
        self.consumer.commit_batch([record])
        self.assertEqual(len(self.fake.commits), 1)
        self.assertFalse(self.fake.commits[0]["asynchronous"])
        self.assertEqual(self.fake.commits[0]["offsets"][0].offset, 11)

    def test_pause_and_resume(self):
        self.consumer.pause()
        self.assertTrue(self.fake.paused)
        self.consumer.resume()
        self.assertTrue(self.fake.resumed)

    def test_revoke_fences_stale_acknowledgement(self):
        record = type("Record", (), {
            "topic": "model-a-results", "partition": 0,
            "offset": 4, "assignment_epoch": self.consumer.rebalance_epoch,
        })()
        self.fake.callbacks["on_revoke"](self.fake, self.fake.assigned)
        with self.assertRaisesRegex(Exception, "stale Kafka commit"):
            self.consumer.commit_batch([record])
        self.assertEqual(self.fake.commits, [])

    def test_lost_partition_fences_commit(self):
        record = type("Record", (), {
            "topic": "model-a-results", "partition": 0,
            "offset": 9, "assignment_epoch": self.consumer.rebalance_epoch,
        })()
        self.fake.callbacks["on_lost"](self.fake, self.fake.assigned)
        with self.assertRaisesRegex(Exception, "stale Kafka commit"):
            self.consumer.commit_batch([record])
        self.assertEqual(self.fake.commits, [])

    def test_unassigned_partition_cannot_be_acknowledged(self):
        record = type("Record", (), {
            "topic": "model-b-results", "partition": 2,
            "offset": 1, "assignment_epoch": self.consumer.rebalance_epoch,
        })()
        with self.assertRaisesRegex(Exception, "unassigned partition"):
            self.consumer.commit_batch([record])

    def test_pause_includes_newly_assigned_partition(self):
        self.consumer.pause()
        self.fake.assigned.append(FakePartition("model-b-results", 1))
        self.consumer.pause()
        self.assertEqual(
            {(p.topic, p.partition) for p in self.fake.paused},
            {("model-a-results", 0), ("model-b-results", 1)},
        )

    def test_paused_consumer_does_not_discard_unexpected_message(self):
        self.fake.messages.append(FakeMessage())
        with self.assertRaisesRegex(Exception, "stopping without acknowledging"):
            self.consumer.service_events(0)
        self.assertEqual(self.fake.commits, [])

    def test_commit_error_propagates(self):
        self.fake.raise_on_commit = RuntimeError("Broker commit unavailable")
        record = type("Record", (), {
            "topic": "model-a-results", "partition": 0, "offset": 1,
            "assignment_epoch": 0,
        })()
        with self.assertRaisesRegex(RuntimeError, "Broker commit unavailable"):
            self.consumer.commit_batch([record])

    def test_close_is_idempotent(self):
        self.consumer.close()
        self.consumer.close()
        self.assertTrue(self.fake.closed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
