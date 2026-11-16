"""Offline regression checks for deterministic Kafka DLT smoke-test startup.

No broker or database connection is required. Run from the repo root:
    python3 -m unittest -v src.results.phase1_dead_letter_smoke_test
"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from confluent_kafka import TopicPartition

from src.kafka.consumer import ConsumerConfig
from src.results.phase1_dead_letter_smoke import (
    SnapshotStartResultConsumer,
    snapshot_source_end_offsets,
)


class FakeMetadataClient:
    def __init__(self):
        self.topic_partitions = {
            "model-a-results": {0: 8, 1: 19},
            "model-b-results": {0: 5},
        }
        self.watermark_calls = []

    def list_topics(self, *, topic, timeout):
        count = self.topic_partitions.get(topic)
        if count is None:
            return SimpleNamespace(topics={})
        return SimpleNamespace(topics={
            topic: SimpleNamespace(
                error=None,
                partitions={index: object() for index in count},
            ),
        })

    def get_watermark_offsets(self, partition, *, timeout, cached):
        self.watermark_calls.append(
            (partition.topic, partition.partition, cached)
        )
        return (0, self.topic_partitions[partition.topic][partition.partition])


class FakeKafkaConsumer:
    def __init__(self, config):
        self.config = config
        self.callbacks = {}
        self.assignments = []
        self._assignment = []
        self.closed = False

    def subscribe(self, topics, **callbacks):
        self.callbacks = callbacks

    def incremental_assign(self, partitions):
        self.assignments.append(partitions)
        self._assignment = partitions

    def assignment(self):
        return self._assignment

    def close(self):
        self.closed = True


class SmokeStartupTest(unittest.TestCase):
    def setUp(self):
        config = ConsumerConfig(
            bootstrap_servers="kafka:9092",
            group_id="throwaway-smoke-group",
            topics=("model-a-results", "model-b-results"),
        )
        self.fake = None

        def factory(kafka_config):
            self.fake = FakeKafkaConsumer(kafka_config)
            return self.fake

        self.consumer = SnapshotStartResultConsumer(
            config,
            starting_offsets={
                ("model-a-results", 0): 8,
                ("model-a-results", 1): 19,
                ("model-b-results", 0): 5,
            },
            consumer_factory=factory,
        )
        self.consumer.subscribe()

    def tearDown(self):
        self.consumer.close()

    def test_metadata_snapshot_uses_uncached_end_offsets(self):
        metadata = FakeMetadataClient()
        offsets = snapshot_source_end_offsets(
            metadata, ("model-a-results", "model-b-results")
        )
        self.assertEqual(offsets, {
            ("model-a-results", 0): 8,
            ("model-a-results", 1): 19,
            ("model-b-results", 0): 5,
        })
        self.assertTrue(all(not cached for _, _, cached in metadata.watermark_calls))

    def test_missing_source_topic_is_error(self):
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            snapshot_source_end_offsets(
                FakeMetadataClient(), ("missing-results",)
            )

    def test_first_assignment_explicitly_uses_snapshot_high_watermarks(self):
        assignments = [
            TopicPartition("model-a-results", 0),
            TopicPartition("model-a-results", 1),
            TopicPartition("model-b-results", 0),
        ]
        self.fake.callbacks["on_assign"](self.fake, assignments)
        result = {
            (p.topic, p.partition): p.offset
            for p in self.fake.assignments[-1]
        }
        self.assertEqual(result, {
            ("model-a-results", 0): 8,
            ("model-a-results", 1): 19,
            ("model-b-results", 0): 5,
        })

    def test_reassigned_partition_not_rewound_to_original_watermark(self):
        assignment = [TopicPartition("model-a-results", 1)]
        callback = self.fake.callbacks["on_assign"]
        callback(self.fake, assignment)
        self.assertEqual(self.fake.assignments[-1][0].offset, 19)
        callback(self.fake, assignment)
        self.assertEqual(
            self.fake.assignments[-1][0].offset,
            TopicPartition("model-a-results", 1).offset,
        )

    def test_unexpected_partition_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Unexpected source partition"):
            self.fake.callbacks["on_assign"](
                self.fake, [TopicPartition("model-b-results", 9)]
            )
        self.assertEqual(self.fake.assignments, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
