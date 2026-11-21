"""Apply a camera YAML file as one immutable PostgreSQL configuration revision.

The YAML is the desired configuration. PostgreSQL cameras.current_config_version
points at the revision that was applied. A running pipeline does not read the
YAML; it reads that revision once when the session starts.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


_FORBIDDEN_KEYS = {
    "password",
    "passwd",
    "secret",
    "token",
    "credential",
    "credentials",
    "connection",
    "connection_string",
    "dsn",
    "username",
    "user",
}


@dataclass(frozen=True, slots=True)
class CameraConfigFile:
    camera_id: str
    zone: str
    region: str | None
    available_models: tuple[str, ...]
    active_models: tuple[str, ...]
    event_storage_root: str

    def as_tuple(self) -> tuple:
        return (
            self.zone,
            self.region,
            self.available_models,
            self.active_models,
            self.event_storage_root,
        )


@dataclass(frozen=True, slots=True)
class ApplyResult:
    camera_id: str
    config_version: int
    created: bool


def load_camera_config_file(path: Path) -> CameraConfigFile:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    forbidden = sorted(set(raw) & _FORBIDDEN_KEYS)
    if forbidden:
        raise ValueError(f"{path} must not contain credential fields: {', '.join(forbidden)}")
    unexpected = sorted(set(raw) - {
        "camera_id", "zone", "region", "available_models", "active_models", "event_storage_root",
    })
    if unexpected:
        raise ValueError(f"{path} has unsupported fields: {', '.join(unexpected)}")

    camera_id = _text(raw.get("camera_id"), "camera_id")
    zone = _text(raw.get("zone"), "zone")
    region = raw.get("region", None)
    if region is not None:
        region = _text(region, "region")
    available = _model_list(raw.get("available_models"), "available_models")
    active = _model_list(raw.get("active_models"), "active_models")
    if any(model not in available for model in active):
        raise ValueError("active_models must be a subset of available_models")
    root = _text(raw.get("event_storage_root"), "event_storage_root")
    if not root.startswith("s3://") or root[5:].find("/") <= 0:
        raise ValueError("event_storage_root must look like s3://bucket/prefix")
    return CameraConfigFile(camera_id, zone, region, available, active, root)


def _execution_constraints() -> tuple[str, frozenset[str]]:
    """The runner executes the detection class named in configs/config.yaml."""
    from configs.settings import load_settings

    tensorrt = load_settings().tensorrt
    return tensorrt.execution_mode, frozenset({tensorrt.detection_class})


def load_applied_camera(database, camera_id: str) -> dict:
    """Read the revision the camera currently points at. No per-frame query."""
    with database.transaction() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT current_config_version, zone, region, event_storage_root, event_bucket,
                   available_models, active_models
            FROM public.camera_current_config
            WHERE camera_id = %s
            """,
            (camera_id,),
        )
        row = cur.fetchone()
    if row is None:
        from configs.paths import ProjectPaths

        paths = ProjectPaths.from_environment()
        config_file = paths.camera_config_file(camera_id)
        raise RuntimeError(
            f"Camera {camera_id!r} has no applied configuration revision. "
            "Apply migration 008_expand_camera_schema.sql and "
            "docker compose run --rm --no-deps -T dev python3 -m tools.apply_camera_config "
            f"{config_file.relative_to(paths.root).as_posix()}"
        )
    version, zone, region, root, event_bucket, available, active = row
    active_models = list(active or [])
    execution_mode, supported_models = _execution_constraints()
    if frozenset(active_models) != supported_models:
        raise ValueError(
            f"Execution mode {execution_mode!r} requires active model "
            f"{next(iter(supported_models))}. "
            f"Camera {camera_id!r} active models are {active_models}. "
            "Additional models may be assigned, but this runner does not execute them."
        )
    return {
        "camera_id": camera_id,
        "execution_mode": execution_mode,
        "event_bucket": event_bucket,
        "config_version": int(version),
        "zone": zone,
        "region": region,
        "available_models": list(available or []),
        "active_models": active_models,
        "event_storage_root": root,
    }


def assignment_identity(available: tuple[str, ...] | list[str], active: tuple[str, ...] | list[str]) -> frozenset:
    """Model membership ignores list order. Active models must be available."""
    active_set = set(active)
    return frozenset((model, model in active_set) for model in available)


def apply_camera_config(connection, desired: CameraConfigFile) -> ApplyResult:
    """Insert a revision only when the normalized configuration differs."""
    if not desired.active_models:
        raise ValueError("active_models must contain at least one model")
    with connection.cursor() as cur:
        if not _table_exists(cur, "camera_config_revision_models"):
            raise RuntimeError(
                "Apply infra/postgres/migrations/008_expand_camera_schema.sql before applying a camera file"
            )
        cur.execute("SELECT set_config('camera_config.apply', 'on', true)")
        cur.execute(
            "SELECT camera_id FROM public.cameras WHERE camera_id = %s FOR UPDATE",
            (desired.camera_id,),
        )
        if cur.fetchone() is None:
            raise RuntimeError(
                f"Camera {desired.camera_id!r} is not registered. "
                "This file does not create cameras or replace names, connections, or credentials."
            )
        cur.execute(
            """
            SELECT r.config_version, r.zone, r.region, r.event_storage_root
            FROM public.cameras AS c
            JOIN public.camera_config_revisions AS r
              ON r.camera_id = c.camera_id
             AND r.config_version = c.current_config_version
            WHERE c.camera_id = %s
            """,
            (desired.camera_id,),
        )
        current = cur.fetchone()
        if current is not None and _same_revision(cur, current, desired):
            return ApplyResult(desired.camera_id, int(current[0]), created=False)

        cur.execute(
            "SELECT COALESCE(MAX(config_version), 0) FROM public.camera_config_revisions WHERE camera_id = %s",
            (desired.camera_id,),
        )
        version = int(cur.fetchone()[0]) + 1
        _insert_revision(cur, desired, version)
        cur.executemany(
            """
            INSERT INTO public.camera_config_revision_models (
                camera_id, config_version, model_id, is_active
            ) VALUES (%s, %s, %s, %s)
            """,
            [
                (desired.camera_id, version, model, model in set(desired.active_models))
                for model in desired.available_models
            ],
        )
        _point_camera_at_revision(cur, desired, version)
    return ApplyResult(desired.camera_id, version, created=True)


def event_config_version(camera_config_version: int) -> int | None:
    """Protobuf field 29 defaults to 0 on messages encoded before it existed."""
    version = int(camera_config_version)
    if version < 0:
        raise ValueError("camera_config_version cannot be negative")
    return None if version == 0 else version


def _same_revision(cursor, row, desired: CameraConfigFile) -> bool:
    _version, zone, region, root = row
    cursor.execute(
        """
        SELECT model_id, is_active
        FROM public.camera_config_revision_models
        WHERE camera_id = %s AND config_version = %s
        """,
        (desired.camera_id, _version),
    )
    stored = frozenset((model, bool(active)) for model, active in cursor.fetchall())
    return (
        zone == desired.zone
        and region == desired.region
        and root == desired.event_storage_root
        and stored == assignment_identity(desired.available_models, desired.active_models)
    )


def _insert_revision(cursor, desired: CameraConfigFile, version: int) -> None:
    columns = _column_names(cursor, "camera_config_revisions")
    payload = {
        "camera_id": desired.camera_id,
        "config_version": version,
        "zone": desired.zone,
        "region": desired.region,
        "event_storage_root": desired.event_storage_root,
    }
    if "available_models" in columns:
        payload["available_models"] = list(desired.available_models)
    if "active_models" in columns:
        payload["active_models"] = list(desired.active_models)
    if "model_assignments" in columns:
        payload["model_assignments"] = "[]"
    names = [name for name in payload if name in columns]
    values = [payload[name] for name in names]
    placeholders = ["%s::jsonb" if name == "model_assignments" else "%s" for name in names]
    cursor.execute(
        f"INSERT INTO public.camera_config_revisions ({', '.join(names)}) "
        f"VALUES ({', '.join(placeholders)})",
        values,
    )


def _point_camera_at_revision(cursor, desired: CameraConfigFile, version: int) -> None:
    columns = _column_names(cursor, "cameras")
    assignments = {
        "current_config_version": version,
        "zone": desired.zone,
        "region": desired.region,
        "available_models": list(desired.available_models),
        "event_storage_root": desired.event_storage_root,
    }
    if "current_model" in columns:
        assignments["current_model"] = (
            desired.active_models[0] if len(desired.active_models) == 1 else None
        )
    sets = [name for name in assignments if name in columns]
    cursor.execute(
        f"UPDATE public.cameras SET {', '.join(f'{name} = %s' for name in sets)} WHERE camera_id = %s",
        [assignments[name] for name in sets] + [desired.camera_id],
    )


def _table_exists(cursor, table: str) -> bool:
    cursor.execute("SELECT to_regclass(%s)", (f"public.{table}",))
    return cursor.fetchone()[0] is not None


def _column_names(cursor, table: str) -> set[str]:
    cursor.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (table,),
    )
    return {row[0] for row in cursor.fetchall()}


def _text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _model_list(value, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    models = tuple(_text(item, name) for item in value)
    if len(set(models)) != len(models):
        raise ValueError(f"{name} contains a duplicate model")
    return models
