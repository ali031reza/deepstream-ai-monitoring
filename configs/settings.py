"""Typed, validated configuration for the DeepStream/Kafka application.

YAML stores non-secret defaults. A local .env is loaded without overriding
variables supplied by Docker, systemd or the operator's shell.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv
import yaml

from configs.paths import ProjectPaths


def _section(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return value


def _positive(value: Any, label: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
        raise ValueError(f"{label} must be {'nonnegative' if allow_zero else 'positive'} integer")
    return value


@dataclass(frozen=True, slots=True)
class SourceSettings:
    mode: str
    camera_id: str
    uri: str
    realtime: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"file", "rtsp"}:
            raise ValueError("pipeline.source.mode must be file or rtsp")
        if not self.camera_id or not self.uri:
            raise ValueError("pipeline.source.camera_id and uri are required")

    def resolved_uri(self, paths: ProjectPaths) -> str:
        return str(paths.resolve(self.uri)) if self.mode == "file" else self.uri


@dataclass(frozen=True, slots=True)
class ModelSettings:
    enabled: bool
    model_id: str
    workers: int
    input_size: int
    hidden_units: int
    top_k: int
    seed: int
    mock_delay_ms: int

    def __post_init__(self) -> None:
        for key in ("workers", "input_size", "hidden_units", "top_k"):
            _positive(getattr(self, key), f"models.{key}")
        _positive(self.mock_delay_ms, "models.mock_delay_ms", allow_zero=True)


@dataclass(frozen=True, slots=True)
class YoloSettings:
    """Runtime settings for the standalone pretrained YOLO11n experiment."""

    model_id: str
    source: Path
    weights: Path
    gpu_id: int
    image_size: int
    confidence: float
    iou: float
    classes: tuple[int, ...]
    quantize: int
    max_detections: int
    output_directory: Path
    run_name: str
    save_video: bool = True

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("yolo.model_id must not be empty")
        _positive(self.gpu_id, "yolo.gpu_id", allow_zero=True)
        _positive(self.image_size, "yolo.image_size")
        _positive(self.max_detections, "yolo.max_detections")
        for name, value in (("confidence", self.confidence), ("iou", self.iou)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"yolo.{name} must be between 0 and 1")
        if not self.classes:
            raise ValueError("yolo.classes must contain at least one class ID")
        for class_id in self.classes:
            _positive(class_id, "yolo.classes[]", allow_zero=True)
        if self.quantize not in {16, 32}:
            raise ValueError("yolo.quantize must be 16 (FP16) or 32 (FP32)")
        if not isinstance(self.save_video, bool):
            raise ValueError("yolo.save_video must be a boolean")
        if not self.run_name.strip() or "/" in self.run_name or "\\" in self.run_name:
            raise ValueError("yolo.run_name must be a nonempty directory name")


@dataclass(frozen=True, slots=True)
class QueueSettings:
    model_input_capacity: int
    overflow_policy: str

    def __post_init__(self) -> None:
        _positive(self.model_input_capacity, "queues.model_input_capacity")
        if self.overflow_policy != "block":
            raise ValueError("Only queues.overflow_policy=block is implemented by the current GPU dispatcher")


@dataclass(frozen=True, slots=True)
class EventSettings:
    enabled: bool
    root: Path
    jpeg_quality: int
    minimum_confidence: float
    every_nth_frame: int
    class_name: str | None
    trigger_policy: str

    def __post_init__(self) -> None:
        if not 1 <= self.jpeg_quality <= 95:
            raise ValueError("event_frames.jpeg_quality must be between 1 and 95")
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("event_frames.minimum_confidence must be between 0 and 1")
        _positive(self.every_nth_frame, "event_frames.every_nth_frame")
        if self.trigger_policy != "any_model_event":
            raise ValueError("Only event_frames.trigger_policy=any_model_event is implemented")


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    host: str
    port: int
    dbname: str
    user: str
    password_env: str
    connect_timeout: int

    def connection_kwargs(self) -> dict[str, Any]:
        # Use the application-role secret only. POSTGRES_PASSWORD is commonly
        # the PostgreSQL bootstrap/superuser password and must not silently be
        # reused to authenticate as DATABASE_USER (normally ai_app).
        secret = os.environ.get(self.password_env)
        if not secret:
            raise ValueError(f"Database password missing: set {self.password_env} in .env or environment")
        return {
            "host": os.environ.get("DATABASE_HOST", self.host),
            "port": int(os.environ.get("DATABASE_PORT", self.port)),
            "dbname": os.environ.get("DATABASE_NAME", self.dbname),
            "user": os.environ.get("DATABASE_USER", self.user),
            "password": secret,
            "connect_timeout": self.connect_timeout,
        }


@dataclass(frozen=True, slots=True)
class KafkaSettings:
    bootstrap_servers: str
    topics: dict[str, str]
    options: dict[str, Any] = field(repr=False)

    @classmethod
    def from_mapping(cls, config: dict[str, Any]) -> "KafkaSettings":
        raw = deepcopy(config)
        bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", raw.get("bootstrap_servers", ""))
        if not isinstance(bootstrap, str) or not bootstrap.strip():
            raise ValueError("kafka.bootstrap_servers must be a nonempty string")
        raw["bootstrap_servers"] = bootstrap.strip()
        topics = _section(raw, "topics")
        overrides = {"mock_model_a": "MODEL_A_RESULT_TOPIC", "mock_model_b": "MODEL_B_RESULT_TOPIC"}
        for model_id in overrides:
            topic = os.environ.get(overrides[model_id], topics.get(model_id, ""))
            if not isinstance(topic, str) or not topic.strip():
                raise ValueError(f"Missing Kafka result topic for {model_id}")
            topics[model_id] = topic.strip()
        if len(set(topics.values())) != len(topics):
            raise ValueError("Kafka result topics must be unique")
        raw["topics"] = topics
        pub = _section(raw, "publisher")
        for key in ("queue_maxsize", "max_batch_messages", "batch_flush_interval_ms"):
            _positive(pub.get(key), f"kafka.publisher.{key}")
        for key in ("enqueue_timeout_seconds", "produce_retry_seconds", "shutdown_timeout_seconds"):
            value = pub.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"kafka.publisher.{key} must be positive")
        _section(pub, "producer")
        for key in ("results_consumer", "batch_processing", "dead_letter", "kafka_admin"):
            _section(raw, key)
        return cls(raw["bootstrap_servers"], topics, raw)

    def as_legacy_dict(self, database: DatabaseSettings) -> dict[str, Any]:
        """Keep existing Kafka publisher and consumer interfaces stable."""
        result = deepcopy(self.options)
        result["database"] = {
            "host": database.host, "port": database.port,
            "dbname": database.dbname, "user": database.user,
            "password_env": database.password_env,
            "connect_timeout": database.connect_timeout,
        }
        return result


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    load_frames: int
    target_fps: tuple[float, ...]
    worker_profiles: tuple[str, ...]
    report_directory: Path
    database_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class AppSettings:
    paths: ProjectPaths
    source: SourceSettings
    gpu_id: int
    sample_limit: int
    models: dict[str, ModelSettings]
    yolo: YoloSettings
    queues: QueueSettings
    events: EventSettings
    kafka: KafkaSettings
    database: DatabaseSettings
    benchmark: BenchmarkSettings
    resource_interval_seconds: float
    metrics_directory: Path

    def legacy_kafka_config(self) -> dict[str, Any]:
        return self.kafka.as_legacy_dict(self.database)


@dataclass(slots=True)
class ConfigurationLoader:
    paths: ProjectPaths = field(default_factory=ProjectPaths.from_environment)

    def load(self, config_path: str | Path | None = None) -> AppSettings:
        load_dotenv(self.paths.env_file, override=False)
        path = self.paths.resolve(config_path or os.environ.get("APP_CONFIG_PATH") or self.paths.config_file)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        if not isinstance(raw, dict):
            raise ValueError("Application configuration root must be a YAML mapping")
        pipeline = _section(raw, "pipeline")
        source = SourceSettings(**_section(pipeline, "source"))
        models_data = _section(raw, "models")
        if set(models_data) != {"model_a", "model_b"}:
            raise ValueError("models must define model_a and model_b")
        models = {name: ModelSettings(**_section(models_data, name)) for name in models_data}
        enabled_models = [model for model in models.values() if model.enabled]
        if len(enabled_models) != 1:
            raise ValueError("This branch requires exactly one enabled model in models")
        yolo_data = _section(raw, "yolo")
        class_ids = yolo_data.get("classes", [0])
        if not isinstance(class_ids, list):
            raise ValueError("yolo.classes must be a YAML list of class IDs")
        yolo = YoloSettings(
            model_id=yolo_data["model_id"],
            source=self.paths.resolve(yolo_data["source"]),
            weights=self.paths.resolve(yolo_data["weights"]),
            gpu_id=_positive(yolo_data["gpu_id"], "yolo.gpu_id", allow_zero=True),
            image_size=_positive(yolo_data["image_size"], "yolo.image_size"),
            confidence=float(yolo_data["confidence"]),
            iou=float(yolo_data["iou"]),
            classes=tuple(class_ids),
            quantize=_positive(yolo_data["quantize"], "yolo.quantize"),
            max_detections=_positive(yolo_data["max_detections"], "yolo.max_detections"),
            output_directory=self.paths.resolve(yolo_data["output_directory"]),
            run_name=yolo_data["run_name"],
            save_video=yolo_data.get("save_video", True),
        )
        queue = QueueSettings(**_section(raw, "queues"))
        event_data = _section(raw, "event_frames")
        events = EventSettings(
            enabled=event_data["enabled"],
            root=self.paths.resolve(event_data["root"]),
            jpeg_quality=event_data["jpeg_quality"],
            minimum_confidence=float(event_data["minimum_confidence"]),
            every_nth_frame=event_data["every_nth_frame"],
            class_name=event_data.get("class_name"),
            trigger_policy=event_data["trigger_policy"],
        )
        db = DatabaseSettings(**_section(raw, "database"))
        kafka = KafkaSettings.from_mapping(_section(raw, "kafka"))
        for model in models.values():
            if model.model_id not in kafka.topics:
                raise ValueError(f"Kafka topic missing for model {model.model_id}")
        benchmark_data = _section(raw, "benchmark")
        benchmark = BenchmarkSettings(
            load_frames=_positive(benchmark_data["load_frames"], "benchmark.load_frames"),
            target_fps=tuple(float(n) for n in benchmark_data["target_fps"]),
            worker_profiles=tuple(benchmark_data["worker_profiles"]),
            report_directory=self.paths.resolve(benchmark_data["report_directory"]),
            database_timeout_seconds=float(benchmark_data["database_timeout_seconds"]),
        )
        monitoring = _section(raw, "monitoring")
        interval = float(monitoring["resource_interval_seconds"])
        if interval <= 0:
            raise ValueError("monitoring.resource_interval_seconds must be positive")
        return AppSettings(
            paths=self.paths, source=source,
            gpu_id=_positive(pipeline.get("gpu_id", 0), "pipeline.gpu_id", allow_zero=True),
            sample_limit=_positive(pipeline.get("sample_limit", 5), "pipeline.sample_limit", allow_zero=True),
            models=models, yolo=yolo, queues=queue, events=events,
            kafka=kafka, database=db, benchmark=benchmark,
            resource_interval_seconds=interval,
            metrics_directory=self.paths.resolve(monitoring["metrics_directory"]),
        )


def load_settings(config_path: str | Path | None = None) -> AppSettings:
    """Backwards-compatible simple facade; prefer ConfigurationLoader in services."""
    return ConfigurationLoader().load(config_path)
