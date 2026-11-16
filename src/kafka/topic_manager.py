
"""Kafka topic creation and verification utilities."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Iterable

from confluent_kafka.admin import AdminClient, NewTopic


DEFAULT_TOPICS = (
    "model-a-results",
    "model-b-results",
    "inference-dead-letter",
)


@dataclass(frozen=True)
class TopicSpec:
    name: str
    partitions: int = 3
    replication_factor: int = 1

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Topic name cannot be empty")

        if self.partitions < 1:
            raise ValueError("Partitions must be positive")

        if self.replication_factor < 1:
            raise ValueError("Replication factor must be positive")


class TopicManager:
    """Create and verify Kafka topics without destructive changes."""

    def __init__(
        self,
        bootstrap_servers: str,
        topic_specs: Iterable[TopicSpec],
        request_timeout_seconds: float = 15.0,
    ) -> None:
        if not bootstrap_servers.strip():
            raise ValueError("bootstrap_servers cannot be empty")

        if request_timeout_seconds <= 0:
            raise ValueError("Request timeout must be positive")

        self.topic_specs = tuple(topic_specs)
        if not self.topic_specs:
            raise ValueError("At least one topic must be configured")

        names = [spec.name for spec in self.topic_specs]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate topic names are not allowed")

        self.request_timeout_seconds = request_timeout_seconds

        self.admin = AdminClient(
            {
                "bootstrap.servers": bootstrap_servers,
                "client.id": "deepstream-topic-manager",
            }
        )

    def ensure_topics(self) -> dict[str, dict]:
        """Create missing topics and verify all configured topics."""

        metadata = self.admin.list_topics(
            timeout=self.request_timeout_seconds
        )

        missing_specs = [
            spec
            for spec in self.topic_specs
            if spec.name not in metadata.topics
        ]

        if missing_specs:
            new_topics = [
                NewTopic(
                    spec.name,
                    spec.partitions,
                    spec.replication_factor,
                )
                for spec in missing_specs
            ]

            futures = self.admin.create_topics(
                new_topics,
                request_timeout=self.request_timeout_seconds,
            )

            for topic_name, future in futures.items():
                try:
                    future.result(
                        timeout=self.request_timeout_seconds
                    )
                    print(f"[CREATED] {topic_name}")

                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to create topic {topic_name}: {exc}"
                    ) from exc

        return self.verify_topics()

    def verify_topics(self) -> dict[str, dict]:
        """Verify partition count, replication and partition leaders."""

        metadata = self.admin.list_topics(
            timeout=self.request_timeout_seconds
        )

        verified: dict[str, dict] = {}

        for spec in self.topic_specs:
            topic_metadata = metadata.topics.get(spec.name)

            if topic_metadata is None:
                raise RuntimeError(
                    f"Topic does not exist: {spec.name}"
                )

            if topic_metadata.error is not None:
                raise RuntimeError(
                    f"Kafka metadata error for {spec.name}: "
                    f"{topic_metadata.error}"
                )

            partitions = topic_metadata.partitions

            actual_partition_count = len(partitions)

            if actual_partition_count != spec.partitions:
                raise ValueError(
                    f"{spec.name}: expected {spec.partitions} partitions, "
                    f"found {actual_partition_count}"
                )

            replication_counts = {
                len(partition.replicas)
                for partition in partitions.values()
            }

            if replication_counts != {spec.replication_factor}:
                raise ValueError(
                    f"{spec.name}: expected replication factor "
                    f"{spec.replication_factor}, "
                    f"found {replication_counts}"
                )

            partition_details = []

            for partition_id, partition in sorted(partitions.items()):
                if partition.leader < 0:
                    raise RuntimeError(
                        f"{spec.name} partition {partition_id} "
                        "has no active leader"
                    )

                partition_details.append(
                    {
                        "partition": partition_id,
                        "leader": partition.leader,
                        "replicas": len(partition.replicas),
                        "in_sync_replicas": len(partition.isrs),
                    }
                )

            verified[spec.name] = {
                "partitions": actual_partition_count,
                "replication_factor": spec.replication_factor,
                "partition_details": partition_details,
            }

        return verified


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manage DeepStream Kafka topics"
    )

    parser.add_argument(
        "--bootstrap-servers",
        default=os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS",
            "127.0.0.1:29092",
        ),
    )

    parser.add_argument(
        "--partitions",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--replication-factor",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--verify-only",
        action="store_true",
    )

    args = parser.parse_args()

    specs = [
        TopicSpec(
            name=topic,
            partitions=args.partitions,
            replication_factor=args.replication_factor,
        )
        for topic in DEFAULT_TOPICS
    ]

    manager = TopicManager(
        bootstrap_servers=args.bootstrap_servers,
        topic_specs=specs,
    )

    print("========== KAFKA TOPIC MANAGER ==========")
    print(f"Bootstrap servers: {args.bootstrap_servers}")

    if args.verify_only:
        results = manager.verify_topics()
    else:
        results = manager.ensure_topics()

    print("\n========== VERIFIED TOPICS ==========")

    for topic_name, details in results.items():
        print(f"\nTopic: {topic_name}")
        print(f"Partitions: {details['partitions']}")
        print(
            "Replication factor: "
            f"{details['replication_factor']}"
        )

        for partition in details["partition_details"]:
            print(
                f"  Partition {partition['partition']}: "
                f"leader={partition['leader']}, "
                f"replicas={partition['replicas']}, "
                f"ISR={partition['in_sync_replicas']}"
            )

    print("\nKAFKA TOPIC VERIFICATION PASSED")


if __name__ == "__main__":
    main()
