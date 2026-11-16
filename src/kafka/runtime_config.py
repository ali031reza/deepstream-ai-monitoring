from __future__ import annotations

import json
import os
from pathlib import Path

from src.kafka.consumer import ConsumerConfig


def default_config_path():
    return (
        Path(__file__).resolve().parents[2]
        / "config"
        / "kafka.json"
    )


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_runtime_config(config_path=None):
    path = Path(config_path) if config_path else default_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"Kafka configuration not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if not isinstance(config, dict):
        raise ValueError("Kafka configuration root must be an object")

    bootstrap = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        config.get("bootstrap_servers", ""),
    )
    if not isinstance(bootstrap, str) or not bootstrap.strip():
        raise ValueError("Kafka bootstrap servers are required")

    topics = config.get("topics", {})
    if not isinstance(topics, dict) or not topics:
        raise ValueError("topics must be a non-empty object")

    env_names = {
        "mock_model_a": "MODEL_A_RESULT_TOPIC",
        "mock_model_b": "MODEL_B_RESULT_TOPIC",
    }
    topic_model_map = {}

    for model_id, configured_topic in topics.items():
        env_name = env_names.get(model_id)
        topic = (
            os.getenv(env_name, configured_topic)
            if env_name
            else configured_topic
        )
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError(f"Missing result topic for {model_id}")
        topic = topic.strip()
        if topic in topic_model_map:
            raise ValueError(f"Duplicate result topic: {topic}")
        topic_model_map[topic] = model_id

    consumer = config.get("results_consumer", {})
    consumer_config = ConsumerConfig(
        bootstrap_servers=bootstrap.strip(),
        group_id=os.getenv(
            "RESULTS_CONSUMER_GROUP_ID",
            consumer.get("group_id", "deepstream-results-processor"),
        ),
        topics=tuple(topic_model_map),
        auto_offset_reset=os.getenv(
            "KAFKA_AUTO_OFFSET_RESET",
            consumer.get("auto_offset_reset", "earliest"),
        ),
        poll_timeout_seconds=float(
            consumer.get("poll_timeout_seconds", 0.5)
        ),
        session_timeout_ms=int(
            consumer.get("session_timeout_ms", 45000)
        ),
        max_poll_interval_ms=int(
            consumer.get("max_poll_interval_ms", 900000)
        ),
    )

    batch = config.get("batch_processing", {})
    retry_attempts = batch.get("retry_attempts", 3)
    if (
        isinstance(retry_attempts, bool)
        or not isinstance(retry_attempts, int)
        or retry_attempts < 0
    ):
        raise ValueError("retry_attempts must be a non-negative integer")

    shutdown_timeout = float(
        batch.get("shutdown_timeout_seconds", 180.0)
    )
    if shutdown_timeout <= 0:
        raise ValueError("shutdown_timeout_seconds must be positive")

    batch_config = {
        "batch_size": _positive_int(
            batch.get("batch_size", 100), "batch_size"
        ),
        "queue_capacity": _positive_int(
            batch.get("queue_capacity", 300), "queue_capacity"
        ),
        "flush_interval_ms": _positive_int(
            batch.get("flush_interval_ms", 50), "flush_interval_ms"
        ),
        "retry_attempts": retry_attempts,
        "retry_backoff_ms": _positive_int(
            batch.get("retry_backoff_ms", 250), "retry_backoff_ms"
        ),
        "shutdown_timeout_seconds": shutdown_timeout,
    }

    dead_letter = config.get("dead_letter", {})
    dead_letter_topic = os.getenv(
        "INFERENCE_DEAD_LETTER_TOPIC",
        dead_letter.get("topic", "inference-dead-letter"),
    )
    if not isinstance(dead_letter_topic, str) or not dead_letter_topic.strip():
        raise ValueError("Dead-letter topic is required")

    dead_letter_flush_timeout = float(
        dead_letter.get("flush_timeout_seconds", 15.0)
    )
    if dead_letter_flush_timeout <= 0:
        raise ValueError("dead-letter flush timeout must be positive")

    database = config.get("database", {})
    password_env = database.get("password_env", "DATABASE_PASSWORD")
    password = os.getenv(password_env) or os.getenv("POSTGRES_PASSWORD")
    if not password:
        raise ValueError(f"Database password is missing. Set {password_env}.")

    database_config = {
        "host": os.getenv(
            "DATABASE_HOST",
            database.get("host", "host.docker.internal"),
        ),
        "port": int(os.getenv(
            "DATABASE_PORT", database.get("port", 5432)
        )),
        "dbname": os.getenv(
            "DATABASE_NAME", database.get("dbname", "ai_monitoring")
        ),
        "user": os.getenv(
            "DATABASE_USER", database.get("user", "ai_app")
        ),
        "password": password,
        "connect_timeout": int(database.get("connect_timeout", 5)),
    }

    producer_config = dict(
        config.get("publisher", {}).get("producer", {})
    )
    producer_config.update({
        "bootstrap.servers": bootstrap.strip(),
        "client.id": "deepstream-dead-letter-publisher",
        "acks": "all",
        "enable.idempotence": True,
    })

    return {
        "bootstrap_servers": bootstrap.strip(),
        "topic_model_map": topic_model_map,
        "consumer_config": consumer_config,
        "batch": batch_config,
        "database": database_config,
        "dead_letter_topic": dead_letter_topic.strip(),
        "dead_letter_flush_timeout_seconds": dead_letter_flush_timeout,
        "dead_letter_producer_config": producer_config,
        "kafka_admin": config.get("kafka_admin", {}),
    }
