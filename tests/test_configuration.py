"""CPU-only tests for the central configuration and path registry."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from configs.paths import ProjectPaths
from configs.settings import ConfigurationLoader, load_settings
from src.results.kafka_config import (
    default_config_path,
    load_kafka_config,
    producer_config_for_model,
    topic_for_model,
)


def test_repository_root_paths_are_independent_of_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    paths = ProjectPaths.from_environment()
    assert paths.root.name == "deepstream-ai-monitoring" or (paths.root / "configs").exists()
    assert paths.resolve("artifacts/event_frames") == paths.root / "artifacts" / "event_frames"
    assert paths.resolve(tmp_path / "absolute") == tmp_path / "absolute"
    assert default_config_path() == paths.config_file


def test_load_typed_sections_from_single_yaml(monkeypatch):
    monkeypatch.delenv("APP_CONFIG_PATH", raising=False)
    settings = load_settings()
    assert settings.models["model_a"].model_id == "mock_model_a"
    assert settings.models["model_b"].model_id == "mock_model_b"
    assert settings.queues.model_input_capacity > 0
    assert settings.events.root == settings.paths.resolve("artifacts/event_frames")
    assert settings.benchmark.report_directory.is_absolute()
    assert settings.source.resolved_uri(settings.paths) == str(settings.paths.resolve(settings.source.uri))


def test_kafka_adapters_retain_existing_contract(monkeypatch):
    monkeypatch.setenv("KAFKA_BOOTSTRAP_SERVERS", "broker.example:19092")
    monkeypatch.setenv("MODEL_A_RESULT_TOPIC", "events-model-a")
    config = load_kafka_config()
    assert config["bootstrap_servers"] == "broker.example:19092"
    assert topic_for_model(config, "mock_model_a") == "events-model-a"
    producer = producer_config_for_model(config, "mock_model_a")
    assert producer["bootstrap.servers"] == "broker.example:19092"
    assert producer["enable.idempotence"] is True
    assert "database" in config


def test_explicit_legacy_json_still_supported(tmp_path):
    settings = load_settings()
    path = tmp_path / "old-config.json"
    path.write_text(json.dumps(settings.legacy_kafka_config()), encoding="utf-8")
    assert topic_for_model(load_kafka_config(path), "mock_model_b") == settings.kafka.topics["mock_model_b"]


def test_invalid_queue_config_is_rejected(tmp_path):
    source = ProjectPaths.from_environment().config_file
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["queues"]["model_input_capacity"] = -1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="queues.model_input_capacity"):
        ConfigurationLoader().load(path)


def test_duplicate_kafka_topics_are_rejected(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_A_RESULT_TOPIC", raising=False)
    monkeypatch.delenv("MODEL_B_RESULT_TOPIC", raising=False)
    source = ProjectPaths.from_environment().config_file
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["kafka"]["topics"]["mock_model_b"] = raw["kafka"]["topics"]["mock_model_a"]
    path = tmp_path / "bad-topics.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="topics must be unique"):
        ConfigurationLoader().load(path)


def test_only_supported_queue_policy_is_accepted(tmp_path):
    raw = yaml.safe_load(ProjectPaths.from_environment().config_file.read_text(encoding="utf-8"))
    raw["queues"]["overflow_policy"] = "drop_newest"
    path = tmp_path / "unsupported.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Only queues.overflow_policy=block"):
        ConfigurationLoader().load(path)


def test_single_model_branch_enables_exactly_one_model(monkeypatch):
    monkeypatch.delenv("APP_CONFIG_PATH", raising=False)
    settings = load_settings()
    enabled = [name for name, model in settings.models.items() if model.enabled]
    assert enabled == ["model_a"]


def test_configuration_rejects_zero_enabled_models(tmp_path):
    source = ProjectPaths.from_environment().config_file
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["models"]["model_a"]["enabled"] = False
    raw["models"]["model_b"]["enabled"] = False
    path = tmp_path / "no-model.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one enabled model"):
        ConfigurationLoader().load(path)


def test_configuration_rejects_two_enabled_models(tmp_path):
    source = ProjectPaths.from_environment().config_file
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["models"]["model_a"]["enabled"] = True
    raw["models"]["model_b"]["enabled"] = True
    path = tmp_path / "two-models.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one enabled model"):
        ConfigurationLoader().load(path)
