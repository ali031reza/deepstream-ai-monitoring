"""Canonical Kafka configuration adapter.

The main application configuration is configs/config.yaml. Explicit legacy
JSON files are accepted as an import format, but no JSON copy is required.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from configs.paths import ProjectPaths
from configs.settings import KafkaSettings, load_settings


MODEL_IDS = ("mock_model_a", "mock_model_b")


def default_config_path() -> Path:
    return ProjectPaths.from_environment().config_file


def load_kafka_config(config_path: str | Path | None = None) -> dict[str, Any]:
    paths = ProjectPaths.from_environment()
    load_dotenv(paths.env_file, override=False)
    path = paths.resolve(config_path) if config_path else paths.resolve(
        os.environ.get("APP_CONFIG_PATH") or paths.config_file
    )
    if path.suffix.lower() == ".json":
        if not path.is_file():
            raise FileNotFoundError(f"Kafka configuration not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict):
            raise ValueError("Kafka configuration root must be a JSON object")
        kafka = KafkaSettings.from_mapping(config)
        result = dict(kafka.options)
        # Legacy files may include a standalone database section.
        if "database" in config:
            result["database"] = config["database"]
        return result
    return load_settings(path).legacy_kafka_config()


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
