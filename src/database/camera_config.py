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


def load_applied_camera(database, camera_id: str) -> dict:
    """Read the revision the camera currently points at. No per-frame query."""
    with database.transaction() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.current_model, c.event_bucket, r.config_version, r.zone, r.region,
                   r.available_models, r.active_models, r.event_storage_root
            FROM public.cameras AS c
            JOIN public.camera_config_revisions AS r
              ON r.camera_id = c.camera_id
             AND r.config_version = c.current_config_version
            WHERE c.camera_id = %s
            """,
            (camera_id,),
        )
        row = cur.fetchone()
    if row is None:
        raise RuntimeError(
            f"Camera {camera_id!r} has no applied configuration revision. "
            "Apply migration 006_camera_config_revisions.sql and "
            "python -m tools.apply_camera_config infra/postgres/cameras_counfig/camera-01.yaml"
        )
    current_model, event_bucket, version, zone, region, available, active, root = row
    if current_model != "person" or "person" not in list(active):
        raise ValueError(f"Camera {camera_id!r} must have active model person")
    return {
        "camera_id": camera_id,
        "current_model": current_model,
        "event_bucket": event_bucket,
        "config_version": int(version),
        "zone": zone,
        "region": region,
        "available_models": list(available),
        "active_models": list(active),
        "event_storage_root": root,
    }


def apply_camera_config(connection, desired: CameraConfigFile) -> ApplyResult:
    """Insert a revision only when the desired values differ. One transaction."""
    with connection.cursor() as cur:
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
            SELECT r.config_version, r.zone, r.region, r.available_models, r.active_models,
                   r.event_storage_root
            FROM public.cameras AS c
            JOIN public.camera_config_revisions AS r
              ON r.camera_id = c.camera_id
             AND r.config_version = c.current_config_version
            WHERE c.camera_id = %s
            """,
            (desired.camera_id,),
        )
        current = cur.fetchone()
        if current is not None and _same_revision(current, desired):
            return ApplyResult(desired.camera_id, int(current[0]), created=False)

        cur.execute(
            "SELECT COALESCE(MAX(config_version), 0) FROM public.camera_config_revisions WHERE camera_id = %s",
            (desired.camera_id,),
        )
        version = int(cur.fetchone()[0]) + 1
        assignments = _assignment_snapshot(cur, desired.camera_id)
        if len(desired.active_models) != 1:
            raise ValueError(
                "cameras.current_model stores one active model; active_models must contain exactly one model"
            )
        sole_active = desired.active_models[0]
        cur.execute(
            """
            INSERT INTO public.camera_config_revisions (
                camera_id, config_version, zone, region, available_models, active_models,
                event_storage_root, model_assignments
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
            """,
            (
                desired.camera_id, version, desired.zone, desired.region,
                list(desired.available_models), list(desired.active_models),
                desired.event_storage_root, assignments,
            ),
        )
        cur.execute(
            """
            UPDATE public.cameras
            SET zone = %s,
                region = %s,
                available_models = %s,
                current_model = %s,
                event_storage_root = %s,
                current_config_version = %s
            WHERE camera_id = %s
            """,
            (
                desired.zone, desired.region, list(desired.available_models), sole_active,
                desired.event_storage_root, version, desired.camera_id,
            ),
        )
    return ApplyResult(desired.camera_id, version, created=True)


def event_config_version(camera_config_version: int) -> int | None:
    """Protobuf field 29 defaults to 0 on messages encoded before it existed."""
    version = int(camera_config_version)
    if version < 0:
        raise ValueError("camera_config_version cannot be negative")
    return None if version == 0 else version


def _same_revision(row, desired: CameraConfigFile) -> bool:
    _version, zone, region, available, active, root = row
    return (
        zone,
        region,
        tuple(available),
        tuple(active),
        root,
    ) == desired.as_tuple()


def _assignment_snapshot(cursor, camera_id: str) -> str:
    cursor.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'camera_model_assignments'
          AND column_name = 'camera_id'
        """
    )
    if cursor.fetchone() is None:
        return "[]"
    cursor.execute(
        """
        SELECT COALESCE(jsonb_agg(to_jsonb(a)), '[]'::jsonb)::text
        FROM public.camera_model_assignments AS a
        WHERE a.camera_id = %s
        """,
        (camera_id,),
    )
    return cursor.fetchone()[0]


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
