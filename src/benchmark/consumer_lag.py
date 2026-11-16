"""Read-only Kafka topic-partition lag snapshot for the Phase 1 consumer group."""
from __future__ import annotations
import argparse
import json


def snapshot(bootstrap_servers: str, group_id: str, topics: tuple[str, ...]):
    from confluent_kafka import Consumer, TopicPartition
    client = Consumer({
        "bootstrap.servers": bootstrap_servers,
        "group.id": group_id,
        "enable.auto.commit": False,
        "auto.offset.reset": "earliest",
    })
    try:
        meta = client.list_topics(timeout=15)
        partitions = []
        for topic in topics:
            if topic not in meta.topics or meta.topics[topic].error is not None:
                raise RuntimeError(f"Kafka topic is missing or unhealthy: {topic}")
            partitions.extend(TopicPartition(topic, p) for p in meta.topics[topic].partitions)
        committed = client.committed(partitions, timeout=15)
        values = []
        for part, offset in zip(partitions, committed):
            low, high = client.get_watermark_offsets(part, timeout=15, cached=False)
            checkpoint = offset.offset
            # -1001 is OFFSET_INVALID: group not initialized / not committed.
            effective = checkpoint if checkpoint >= 0 else low
            values.append({
                "topic": part.topic, "partition": part.partition,
                "earliest_available_offset": low, "high_watermark": high,
                "committed_next_offset": checkpoint,
                "lag_messages": max(0, high - effective),
            })
        return {"consumer_group": group_id, "total_lag_messages": sum(x["lag_messages"] for x in values),
                "partitions": values}
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/kafka.json")
    parser.add_argument("--group-id", default=None)
    args = parser.parse_args()
    from src.results.kafka_config import load_kafka_config
    config = load_kafka_config(args.config)
    report = snapshot(config["bootstrap_servers"],
                      args.group_id or config["results_consumer"]["group_id"],
                      tuple(config["topics"].values()))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
