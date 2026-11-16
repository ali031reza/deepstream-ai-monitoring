"""Typed runtime configuration for the Kafka -> PostgreSQL result processor."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from src.kafka.consumer import ConsumerConfig
from src.results.kafka_config import default_config_path, load_kafka_config


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeConfiguration:
    bootstrap_servers: str
    topic_model_map: dict[str, str]
    consumer_config: ConsumerConfig
    batch: dict[str, Any]
    database: dict[str, Any]
    dead_letter_topic: str
    dead_letter_flush_timeout_seconds: float
    dead_letter_producer_config: dict[str, Any]
    kafka_admin: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Compatibility with the existing pipeline and result processor."""
        return {
            "bootstrap_servers": self.bootstrap_servers,
            "topic_model_map": self.topic_model_map,
            "consumer_config": self.consumer_config,
            "batch": self.batch,
            "database": self.database,
            "dead_letter_topic": self.dead_letter_topic,
            "dead_letter_flush_timeout_seconds": self.dead_letter_flush_timeout_seconds,
            "dead_letter_producer_config": self.dead_letter_producer_config,
            "kafka_admin": self.kafka_admin,
        }

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "RuntimeConfiguration":
        config = load_kafka_config(config_path)
        bootstrap = config["bootstrap_servers"]
        topics = config["topics"]
        topic_model_map = {topic: model_id for model_id, topic in topics.items()}
        if len(topic_model_map) != len(topics):
            raise ValueError("Duplicate Kafka result topic")

        consumer = config.get("results_consumer", {})
        consumer_config = ConsumerConfig(
            bootstrap_servers=bootstrap,
            group_id=os.getenv("RESULTS_CONSUMER_GROUP_ID", consumer.get("group_id", "deepstream-results-processor")),
            topics=tuple(topic_model_map),
            auto_offset_reset=os.getenv("KAFKA_AUTO_OFFSET_RESET", consumer.get("auto_offset_reset", "earliest")),
            poll_timeout_seconds=float(consumer.get("poll_timeout_seconds", 0.5)),
            session_timeout_ms=int(consumer.get("session_timeout_ms", 45000)),
            max_poll_interval_ms=int(consumer.get("max_poll_interval_ms", 900000)),
        )

        batch_data = config.get("batch_processing", {})
        retries = batch_data.get("retry_attempts", 3)
        if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
            raise ValueError("retry_attempts must be nonnegative integer")
        shutdown = float(batch_data.get("shutdown_timeout_seconds", 180))
        if shutdown <= 0:
            raise ValueError("shutdown_timeout_seconds must be positive")
        batch = {
            "batch_size": _positive_int(batch_data.get("batch_size", 100), "batch_size"),
            "queue_capacity": _positive_int(batch_data.get("queue_capacity", 300), "queue_capacity"),
            "flush_interval_ms": _positive_int(batch_data.get("flush_interval_ms", 50), "flush_interval_ms"),
            "retry_attempts": retries,
            "retry_backoff_ms": _positive_int(batch_data.get("retry_backoff_ms", 250), "retry_backoff_ms"),
            "shutdown_timeout_seconds": shutdown,
        }

        dead_letter = config.get("dead_letter", {})
        topic = os.getenv("INFERENCE_DEAD_LETTER_TOPIC", dead_letter.get("topic", "inference-dead-letter"))
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("Dead-letter topic required")
        flush = float(dead_letter.get("flush_timeout_seconds", 15))
        if flush <= 0:
            raise ValueError("Dead-letter flush timeout must be positive")
        db = config.get("database", {})
        password_env = db.get("password_env", "DATABASE_PASSWORD")
        password = os.getenv(password_env) or os.getenv("POSTGRES_PASSWORD")
        if not password:
            raise ValueError(f"Database password is missing. Set {password_env}.")
        database = {
            "host": os.getenv("DATABASE_HOST", db.get("host", "host.docker.internal")),
            "port": int(os.getenv("DATABASE_PORT", db.get("port", 5432))),
            "dbname": os.getenv("DATABASE_NAME", db.get("dbname", "ai_monitoring")),
            "user": os.getenv("DATABASE_USER", db.get("user", "ai_app")),
            "password": password,
            "connect_timeout": int(db.get("connect_timeout", 5)),
        }
        producer = dict(config.get("publisher", {}).get("producer", {}))
        producer.update({
            "bootstrap.servers": bootstrap,
            "client.id": "deepstream-dead-letter-publisher",
            "acks": "all",
            "enable.idempotence": True,
        })
        return cls(bootstrap, topic_model_map, consumer_config, batch, database,
                   topic.strip(), flush, producer, dict(config.get("kafka_admin", {})))


def load_runtime_config(config_path: str | Path | None = None) -> dict[str, Any]:
    return RuntimeConfiguration.load(config_path).as_dict()
