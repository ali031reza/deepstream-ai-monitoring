"""Create disposable databases, exercise the camera-schema migrations, and drop them.

This never connects to ai_monitoring and never applies migration 009 there.
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "infra" / "postgres" / "migrations"
CONTAINER = "postgres-postgres-1"
EXPAND_DB = "ai_monitoring_schema_test"
CONTRACT_DB = "ai_monitoring_schema_contract"
FRESH_DB = "ai_monitoring_schema_fresh"

HISTORICAL_EVENTS = r"""
INSERT INTO camera_events (
    event_id, camera_id, source_session_id, frame_sequence, model_id, event_type,
    event_at, source_video_ms, event_storage_root,
    clean_key, clean_bytes, clean_sha256, boxed_key, boxed_bytes, boxed_sha256,
    detections, payload_sha256, config_version, event, confidence, person_id
) VALUES
('hist-index-zero', 'camera-01', 'hist-session', 10, 'yolo11n-person', 'person_detected',
 now(), 0, 's3://camera-events/events/camera-01',
 'events/camera-01/hist-session/000000000010.jpg', 10, repeat('a', 64),
 'events/camera-01/hist-session/000000000010.boxed.jpg', 11, repeat('b', 64),
 '[{"class_name":"person","confidence":0.9}]'::jsonb,
 repeat('c', 64), 1, 'person', 0.9, '10:1'),
('hist-tie', 'camera-01', 'hist-session', 11, 'yolo11n-person', 'person_detected',
 now(), 0, 's3://camera-events/events/camera-01',
 'events/camera-01/hist-session/000000000011.jpg', 10, repeat('d', 64),
 'events/camera-01/hist-session/000000000011.boxed.jpg', 11, repeat('e', 64),
 '[{"class_name":"person","confidence":0.8},{"class_name":"fire","confidence":0.99},{"class_name":"person","confidence":0.8}]'::jsonb,
 repeat('f', 64), 1, 'person', 0.8, '11:2'),
('hist-unsorted', 'camera-01', 'hist-session', 12, 'yolo11n-person', 'person_detected',
 now(), 0, 's3://camera-events/events/camera-01',
 'events/camera-01/hist-session/000000000012.jpg', 10, repeat('1', 64),
 'events/camera-01/hist-session/000000000012.boxed.jpg', 11, repeat('2', 64),
 '[{"class_name":"person","confidence":0.66},{"class_name":"person","confidence":0.91}]'::jsonb,
 repeat('3', 64), 1, 'person', 0.91, '12:9'),
('hist-unprovable', 'camera-01', 'hist-session', 40, 'yolo11n-person', 'person_detected',
 now(), 0, 's3://camera-events/events/camera-01',
 'events/camera-01/hist-session/000000000040.jpg', 10, repeat('4', 64),
 'events/camera-01/hist-session/000000000040.boxed.jpg', 11, repeat('5', 64),
 '[]'::jsonb, repeat('6', 64), 1, 'person', 0.8, '6090:1'),
('hist-unversioned', 'camera-01', 'hist-session', 41, 'yolo11n-person', 'person_detected',
 now(), 0, 's3://camera-events/events/camera-01',
 'events/camera-01/hist-session/000000000041.jpg', 10, repeat('7', 64),
 'events/camera-01/hist-session/000000000041.boxed.jpg', 11, repeat('8', 64),
 '[{"class_name":"person","confidence":0.7}]'::jsonb,
 repeat('9', 64), NULL, 'person', 0.7, '41:1');
"""


def _psql(database: str, sql: str, filename: Path | None = None) -> None:
    command = [
        "docker", "exec", "-i", CONTAINER, "sh", "-c",
        'exec psql -v ON_ERROR_STOP=1 -U "${POSTGRES_USER}" -d "$1"',
        "sh", database,
    ]
    if filename is not None:
        sql = filename.read_text(encoding="utf-8")
    completed = subprocess.run(command, input=sql, text=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"psql failed for {database} ({filename or 'statement'})")


def _recreate(database: str) -> None:
    _psql("postgres", f"DROP DATABASE IF EXISTS {database};")
    _psql("postgres", f"CREATE DATABASE {database};")


def _migrate(database: str, numbers: list[str]) -> None:
    for number in numbers:
        path = MIGRATIONS / number
        print(f"apply {number} to {database}")
        _psql(database, "", path)


def _run_tests() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "src.database.camera_schema_migration_test"
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit("camera schema migration tests failed")


def main() -> None:
    os.environ["DATABASE_HOST"] = "127.0.0.1"
    os.environ["PYTHONPATH"] = str(ROOT)
    os.chdir(ROOT)
    base = ["004_camera_events.sql", "005_event_bucket.sql", "006_camera_config_revisions.sql", "007_event_observation.sql"]
    expand = base + ["008_expand_camera_schema.sql"]

    print(f"fresh install {FRESH_DB}")
    _recreate(FRESH_DB)
    _migrate(FRESH_DB, expand + ["009_contract_camera_schema.sql"])
    os.environ["DATABASE_NAME"] = FRESH_DB
    _run_tests()

    print(f"upgrade {EXPAND_DB}")
    _recreate(EXPAND_DB)
    _migrate(EXPAND_DB, base)
    _psql(EXPAND_DB, HISTORICAL_EVENTS)
    _migrate(EXPAND_DB, ["008_expand_camera_schema.sql"])
    os.environ["DATABASE_NAME"] = EXPAND_DB
    _run_tests()

    print(f"contract upgrade {CONTRACT_DB}")
    _recreate(CONTRACT_DB)
    _migrate(CONTRACT_DB, base)
    _psql(CONTRACT_DB, HISTORICAL_EVENTS)
    _migrate(CONTRACT_DB, ["008_expand_camera_schema.sql", "009_contract_camera_schema.sql"])
    os.environ["DATABASE_NAME"] = CONTRACT_DB
    _run_tests()

    for database in (FRESH_DB, EXPAND_DB, CONTRACT_DB):
        _psql("postgres", f"DROP DATABASE IF EXISTS {database};")
    print("disposable camera-schema databases dropped")


if __name__ == "__main__":
    main()
