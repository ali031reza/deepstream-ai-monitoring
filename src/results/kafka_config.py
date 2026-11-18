from __future__ import annotations

from pathlib import Path
from typing import Any


from configs.paths import ProjectPaths
from configs.settings import load_settings


def default_config_path() -> Path:
    return ProjectPaths.from_environment().config_file


def load_kafka_config(config_path: str | Path | None = None) -> dict[str, Any]:
    return load_settings(config_path).legacy_kafka_config()


def topic_for_model(config: dict[str, Any], model_id: str) -> str:
    try:
        return config["topics"][model_id]
    except KeyError as exc:
        raise ValueError(f"No Kafka topic configured for model_id={model_id!r}") from exc


def producer_config_for_model(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    options = dict(config["publisher"].get("producer", {}))
    options["bootstrap.servers"] = config["bootstrap_servers"]
    options["client.id"] = f"deepstream-{model_id}-publisher"
    return options


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
