from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from confluent_kafka.admin import AdminClient, NewPartitions, NewTopic


def _default_config_path():
    return Path(__file__).resolve().parents[2] / "config" / "kafka.json"


class KafkaTopicManager:
    """Create missing topics and increase partition counts without deleting data."""

    def __init__(self, bootstrap_servers, topic_names, admin_config):
        if not bootstrap_servers.strip():
            raise ValueError("bootstrap_servers is required")
        if not topic_names:
            raise ValueError("At least one topic is required")
        if len(set(topic_names)) != len(topic_names):
            raise ValueError("Duplicate topic names are not allowed")

        self.topic_names = tuple(topic_names)
        self.partitions = int(admin_config.get("partitions", 3))
        self.replication_factor = int(
            admin_config.get("replication_factor", 1)
        )
        self.retention_ms = int(
            admin_config.get("retention_ms", 604800000)
        )
        self.timeout = float(
            admin_config.get("request_timeout_seconds", 30)
        )

        if self.partitions < 1 or self.replication_factor < 1:
            raise ValueError("Partitions and replication factor must be positive")

        self.admin = AdminClient({
            "bootstrap.servers": bootstrap_servers,
            "client.id": "deepstream-kafka-topic-manager",
        })

    def ensure_topics(self):
        metadata = self.admin.list_topics(timeout=self.timeout)
        existing = metadata.topics
        new_topics = []
        increases = []

        for name in self.topic_names:
            if name not in existing:
                new_topics.append(NewTopic(
                    name,
                    num_partitions=self.partitions,
                    replication_factor=self.replication_factor,
                    config={"retention.ms": str(self.retention_ms)},
                ))
                continue

            topic_meta = existing[name]
            if topic_meta.error is not None:
                raise RuntimeError(
                    f"Cannot inspect topic {name}: {topic_meta.error}"
                )

            current = len(topic_meta.partitions)
            if current < self.partitions:
                increases.append(NewPartitions(name, self.partitions))
            elif current > self.partitions:
                print(
                    f"Keeping {name} at {current} partitions; "
                    "Kafka cannot decrease partition count."
                )

        if new_topics:
            futures = self.admin.create_topics(
                new_topics,
                request_timeout=self.timeout,
            )
            for name, future in futures.items():
                try:
                    future.result(timeout=self.timeout)
                    print(f"[CREATED] {name}")
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to create topic {name}: {exc}"
                    ) from exc

        if increases:
            futures = self.admin.create_partitions(
                increases,
                request_timeout=self.timeout,
            )
            for name, future in futures.items():
                try:
                    future.result(timeout=self.timeout)
                    print(f"[PARTITIONS INCREASED] {name}")
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to increase partitions for {name}: {exc}"
                    ) from exc

        return self.verify_topics()

    def verify_topics(self):
        metadata = self.admin.list_topics(timeout=self.timeout)
        verified = {}

        for name in self.topic_names:
            topic_meta = metadata.topics.get(name)
            if topic_meta is None:
                raise RuntimeError(f"Topic does not exist: {name}")
            if topic_meta.error is not None:
                raise RuntimeError(
                    f"Kafka metadata error for {name}: {topic_meta.error}"
                )

            partitions = topic_meta.partitions
            replica_counts = {
                len(partition.replicas)
                for partition in partitions.values()
            }

            if len(partitions) < self.partitions:
                raise ValueError(
                    f"{name}: expected at least {self.partitions} partitions, "
                    f"found {len(partitions)}"
                )
            if replica_counts != {self.replication_factor}:
                raise ValueError(
                    f"{name}: expected replication factor "
                    f"{self.replication_factor}, found {replica_counts}"
                )

            for partition_id, partition in partitions.items():
                if partition.leader < 0:
                    raise RuntimeError(
                        f"{name} partition {partition_id} has no active leader"
                    )

            verified[name] = {
                "partitions": len(partitions),
                "replication_factor": self.replication_factor,
            }

        return verified


def main():
    parser = argparse.ArgumentParser(
        description="Create and verify DeepStream Kafka topics"
    )
    parser.add_argument("--config", default=str(_default_config_path()))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    bootstrap = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        config.get("bootstrap_servers", ""),
    )
    topics = list(config.get("topics", {}).values())
    topics.append(
        os.getenv(
            "INFERENCE_DEAD_LETTER_TOPIC",
            config.get("dead_letter", {}).get(
                "topic", "inference-dead-letter"
            ),
        )
    )

    manager = KafkaTopicManager(
        bootstrap,
        topics,
        config.get("kafka_admin", {}),
    )

    result = (
        manager.verify_topics()
        if args.verify_only
        else manager.ensure_topics()
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
