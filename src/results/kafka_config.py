"""Load and validate the standalone Kafka publishing configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


MODEL_IDS = ("mock_model_a", "mock_model_b")


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "kafka.json"


def load_kafka_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else default_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"Kafka configuration not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if not isinstance(config, dict):
        raise ValueError("Kafka configuration root must be a JSON object")

    bootstrap = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS",
        config.get("bootstrap_servers", ""),
    )
    if not isinstance(bootstrap, str) or not bootstrap.strip():
        raise ValueError("bootstrap_servers must be a non-empty string")
    config["bootstrap_servers"] = bootstrap.strip()

    topics = config.get("topics")
    if not isinstance(topics, dict):
        raise ValueError("topics must be an object")

    env_topic_names = {
        "mock_model_a": "MODEL_A_RESULT_TOPIC",
        "mock_model_b": "MODEL_B_RESULT_TOPIC",
    }
    for model_id in MODEL_IDS:
        topic = os.getenv(env_topic_names[model_id], topics.get(model_id, ""))
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError(f"Missing Kafka topic for {model_id}")
        topics[model_id] = topic.strip()

    publisher = config.get("publisher")
    if not isinstance(publisher, dict):
        raise ValueError("publisher must be an object")

    positive_int_fields = ("queue_maxsize", "max_batch_messages", "batch_flush_interval_ms")
    for field in positive_int_fields:
        value = publisher.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"publisher.{field} must be a positive integer")

    positive_number_fields = (
        "enqueue_timeout_seconds",
        "produce_retry_seconds",
        "shutdown_timeout_seconds",
    )
    for field in positive_number_fields:
        value = publisher.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"publisher.{field} must be a positive number")

    producer_config = publisher.get("producer", {})
    if not isinstance(producer_config, dict):
        raise ValueError("publisher.producer must be an object")

    return config


def topic_for_model(config: dict[str, Any], model_id: str) -> str:
    try:
        return config["topics"][model_id]
    except KeyError as exc:
        raise ValueError(f"No Kafka topic configured for model_id={model_id!r}") from exc


def producer_config_for_model(
    config: dict[str, Any],
    model_id: str,
) -> dict[str, Any]:
    producer_config = dict(config["publisher"].get("producer", {}))
    producer_config["bootstrap.servers"] = config["bootstrap_servers"]
    producer_config["client.id"] = f"deepstream-{model_id}-publisher"
    return producer_config


def publisher_options(config: dict[str, Any]) -> dict[str, Any]:
    publisher = config["publisher"]
    return {
        "queue_maxsize": publisher["queue_maxsize"],
        "max_batch_messages": publisher["max_batch_messages"],
        "batch_flush_interval_ms": publisher["batch_flush_interval_ms"],
        "enqueue_timeout_seconds": float(publisher["enqueue_timeout_seconds"]),
        "produce_retry_seconds": float(publisher["produce_retry_seconds"]),
        "shutdown_timeout_seconds": float(publisher["shutdown_timeout_seconds"]),
    }
