"""Checks for the disposable camera-schema databases.

The driver in tools/check_camera_schema_refactor.py creates those databases.
This file refuses to run against any other database name.
"""
from __future__ import annotations

import os
import unittest
import uuid

from configs.settings import load_settings
from src.database.batch_result_repository import DatabaseBatchRepository
from src.database.camera_config import (
    ApplyResult,
    CameraConfigFile,
    apply_camera_config,
    load_camera_config_file,
)
from src.database.client import DatabaseClient
from src.proto.inference_result_pb2 import Detection, InferenceResult


EXPAND_DB = "ai_monitoring_schema_test"
CONTRACT_DB = "ai_monitoring_schema_contract"
FRESH_DB = "ai_monitoring_schema_fresh"
CAMERA_FILE = (
    __import__("pathlib").Path(__file__).resolve().parents[2]
    / "infra"
    / "postgres"
    / "cameras_counfig"
    / "camera-01.yaml"
)


def _database():
    settings = load_settings()
    client = DatabaseClient(**settings.database.connection_kwargs())
    name = client._config["dbname"]
    if name not in {EXPAND_DB, CONTRACT_DB, FRESH_DB}:
        raise unittest.SkipTest(f"refusing to run schema checks against {name}")
    client.check_connection()
    return client, name


def _event_result(*, result_id: str, worker_id: str, frame_sequence: int, model_id: str) -> InferenceResult:
    result = InferenceResult(
        camera_id="camera-01",
        frame_id=str(frame_sequence),
        model_id=model_id,
        processed_at_ms=1_700_000_000_000,
        status="processed",
        is_event=True,
        event_type="person_detected",
        storage_key=f"events/camera-01/replay/{frame_sequence:012d}.jpg",
        image_format="jpeg",
        image_size_bytes=10,
        image_sha256="a" * 64,
        captured_at_ms=1_700_000_000_000,
        source_session_id="replay-session",
        result_id=result_id,
        worker_id=worker_id,
        frame_sequence=frame_sequence,
        schema_version="2.0",
        event_storage_root="s3://camera-events/events/camera-01",
        boxed_key=f"events/camera-01/replay/{frame_sequence:012d}.{model_id}.boxed.jpg",
        boxed_bytes=11,
        boxed_sha256="b" * 64,
        source_video_ms=0,
        camera_config_version=1,
        primary_class="person",
        primary_confidence=0.91,
        engine_digest="d" * 64,
    )
    result.primary_detection_index = 0
    result.detections.append(Detection(
        class_id=0, class_name="person", confidence=0.91,
        bbox_normalized=[0.1, 0.1, 0.4, 0.8],
    ))
    return result


class CameraSchemaMigrationTest(unittest.TestCase):
    def setUp(self):
        self.database, self.dbname = _database()

    def test_upgrade_preserves_history_and_assignment_changes(self):
        if self.dbname != EXPAND_DB:
            self.skipTest("expand database only")
        with self.database.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT event_id, config_version, primary_class, primary_confidence,
                       primary_detection_index, person_id, engine_digest,
                       's3://' || event_bucket || '/' || clean_key,
                       's3://' || event_bucket_stored || '/' || clean_key,
                       's3://' || split_part(event_storage_root, '/', 3) || '/' || clean_key,
                       's3://' || event_bucket || '/' || boxed_key,
                       's3://' || event_bucket_stored || '/' || boxed_key
                FROM camera_events
                WHERE event_id LIKE 'hist-%'
                ORDER BY event_id
                """
            )
            rows = {row[0]: row[1:] for row in cur.fetchall()}
        zero = rows["hist-index-zero"]
        self.assertEqual(zero[1], "person")
        self.assertAlmostEqual(zero[2], 0.9)
        self.assertEqual(zero[3], 0)
        self.assertEqual(zero[4], "10:1")
        self.assertIsNone(zero[5])
        self.assertEqual(rows["hist-tie"][4], "11:2")
        self.assertEqual(rows["hist-tie"][3], 0)
        self.assertEqual(rows["hist-unsorted"][3], 1)
        self.assertEqual(rows["hist-unsorted"][4], "12:9")
        self.assertIsNone(rows["hist-unprovable"][1])
        self.assertIsNone(rows["hist-unprovable"][2])
        self.assertIsNone(rows["hist-unprovable"][3])
        self.assertEqual(rows["hist-unprovable"][4], "6090:1")
        self.assertIsNone(rows["hist-unversioned"][0])
        self.assertEqual(rows["hist-unversioned"][3], 0)
        for row in rows.values():
            bucket_clean, stored_clean, root_clean, bucket_boxed, stored_boxed = row[6:]
            self.assertEqual(bucket_clean, stored_clean)
            self.assertEqual(bucket_clean, root_clean)
            self.assertEqual(bucket_boxed, stored_boxed)
            self.assertTrue(bucket_clean.startswith("s3://camera-events/"))

        with self.database.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT zone, event_bucket, available_models, active_models
                FROM camera_current_config WHERE camera_id = 'camera-01'
                """
            )
            zone, bucket, available, active = cur.fetchone()
        self.assertEqual(zone, "A")
        self.assertEqual(bucket, "camera-events")
        self.assertEqual(list(available), ["fire", "person"])
        self.assertEqual(list(active), ["person"])

        desired = load_camera_config_file(CAMERA_FILE)
        with self.database.transaction() as conn:
            first = apply_camera_config(conn, desired)
            second = apply_camera_config(conn, desired)
            reordered = apply_camera_config(conn, CameraConfigFile(
                camera_id="camera-01", zone="A", region=None,
                available_models=("person", "fire"), active_models=("person",),
                event_storage_root=desired.event_storage_root,
            ))
            changed = apply_camera_config(conn, CameraConfigFile(
                camera_id="camera-01", zone="A", region=None,
                available_models=("fire", "person"), active_models=("fire", "person"),
                event_storage_root=desired.event_storage_root,
            ))
        self.assertFalse(first.created)
        self.assertEqual(first.config_version, second.config_version)
        self.assertEqual(reordered, ApplyResult("camera-01", first.config_version, False))
        self.assertEqual(changed, ApplyResult("camera-01", first.config_version + 1, True))

        with self.database.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.zone,
                       array_agg(m.model_id ORDER BY m.model_id) FILTER (WHERE m.is_active),
                       array_agg(m.model_id ORDER BY m.model_id) FILTER (WHERE NOT m.is_active)
                FROM camera_events AS e
                JOIN camera_config_revisions AS r
                  ON r.camera_id = e.camera_id AND r.config_version = e.config_version
                JOIN camera_config_revision_models AS m
                  ON m.camera_id = r.camera_id AND m.config_version = r.config_version
                WHERE e.event_id = 'hist-index-zero'
                GROUP BY r.zone
                """
            )
            historical_zone, historical_active, historical_inactive = cur.fetchone()
        self.assertEqual(historical_zone, "A")
        self.assertEqual(list(historical_active), ["person"])
        self.assertEqual(list(historical_inactive), ["fire"])

        with self.database.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO camera_events (
                    event_id, camera_id, source_session_id, frame_sequence, model_id, event_type,
                    event_at, source_video_ms, event_storage_root, event_bucket_stored,
                    clean_key, clean_bytes, clean_sha256, boxed_key, boxed_bytes, boxed_sha256,
                    detections, payload_sha256, config_version, primary_class, primary_confidence,
                    primary_detection_index
                ) VALUES
                ('shared-person', 'camera-01', 'shared-session', 7, 'yolo11n-person', 'person_detected',
                 now(), 0, 's3://camera-events/events/camera-01', 'camera-events',
                 'events/camera-01/shared-session/000000000007.jpg', 10, %s,
                 'events/camera-01/shared-session/000000000007.yolo11n-person.boxed.jpg', 11, %s,
                 '[]'::jsonb, %s, 1, 'person', 0.7, 0),
                ('shared-fire', 'camera-01', 'shared-session', 7, 'yolo11n-fire', 'person_detected',
                 now(), 0, 's3://camera-events/events/camera-01', 'camera-events',
                 'events/camera-01/shared-session/000000000007.jpg', 10, %s,
                 'events/camera-01/shared-session/000000000007.yolo11n-fire.boxed.jpg', 12, %s,
                 '[]'::jsonb, %s, 1, 'fire', 0.8, 0)
                """,
                ("a" * 64, "b" * 64, "c" * 64, "a" * 64, "d" * 64, "e" * 64),
            )
            cur.execute(
                """
                SELECT clean_key, boxed_key FROM camera_events
                WHERE source_session_id = 'shared-session' ORDER BY model_id
                """
            )
            stored = cur.fetchall()
        self.assertEqual(stored[0][0], stored[1][0])
        self.assertNotEqual(stored[0][1], stored[1][1])
        self._assert_retry_and_conflict(frame_sequence=80)

    def test_contract_keeps_verified_history_and_accepts_new_events(self):
        if self.dbname != CONTRACT_DB:
            self.skipTest("contract database only")
        with self.database.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = 'camera_events'
                """
            )
            columns = {row[0] for row in cur.fetchall()}
            self.assertNotIn("person_id", columns)
            self.assertNotIn("event_storage_root", columns)
            self.assertNotIn("event", columns)
            self.assertIn("event_bucket", columns)
            self.assertIn("primary_detection_index", columns)
            cur.execute(
                """
                SELECT config_version, primary_detection_index, event_bucket, clean_key, boxed_key
                FROM camera_events WHERE event_id = 'hist-unprovable'
                """
            )
            version, index, bucket, clean_key, boxed_key = cur.fetchone()
            self.assertEqual(version, 1)
            self.assertIsNone(index)
            self.assertEqual(bucket, "camera-events")
            self.assertEqual(
                f"s3://{bucket}/{clean_key}",
                "s3://camera-events/events/camera-01/hist-session/000000000040.jpg",
            )
            self.assertTrue(boxed_key.endswith(".boxed.jpg"))
            cur.execute(
                "SELECT config_version FROM camera_events WHERE event_id = 'hist-unversioned'"
            )
            self.assertIsNone(cur.fetchone()[0])
            cur.execute(
                """
                SELECT zone, event_bucket, active_models
                FROM camera_current_config WHERE camera_id = 'camera-01'
                """
            )
            zone, view_bucket, active = cur.fetchone()
        self.assertEqual(zone, "A")
        self.assertEqual(view_bucket, "camera-events")
        self.assertIn("person", list(active))
        self._assert_retry_and_conflict(frame_sequence=90)

    def test_fresh_install_exposes_the_current_config_and_stores_an_event(self):
        if self.dbname != FRESH_DB:
            self.skipTest("fresh database only")
        desired = load_camera_config_file(CAMERA_FILE)
        with self.database.transaction() as conn:
            first = apply_camera_config(conn, desired)
            second = apply_camera_config(conn, desired)
        self.assertEqual(first, ApplyResult("camera-01", 1, False))
        self.assertEqual(second, ApplyResult("camera-01", 1, False))
        with self.database.transaction() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT camera_id, zone, event_bucket, available_models, active_models
                FROM camera_current_config WHERE camera_id = 'camera-01'
                """
            )
            row = cur.fetchone()
        self.assertEqual(row[0], "camera-01")
        self.assertEqual(row[1], "A")
        self.assertEqual(row[2], "camera-events")
        self.assertEqual(list(row[3]), ["fire", "person"])
        self.assertEqual(list(row[4]), ["person"])
        self._assert_retry_and_conflict(frame_sequence=70)

    def _assert_retry_and_conflict(self, *, frame_sequence: int):
        repository = DatabaseBatchRepository(self.database)
        result_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"schema-retry-{frame_sequence}"))
        original = _event_result(
            result_id=result_id, worker_id="worker-a:1",
            frame_sequence=frame_sequence, model_id="yolo11n-person",
        )
        repository.persist_batch([original])
        repository.persist_batch([original])
        with self.database.transaction() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM camera_events WHERE event_id = %s", (result_id,))
            self.assertEqual(cur.fetchone()[0], 1)
        conflict = _event_result(
            result_id=result_id, worker_id="worker-b:2",
            frame_sequence=frame_sequence, model_id="yolo11n-person",
        )
        with self.assertRaises(ValueError):
            repository.persist_batch([conflict])
        with self.database.transaction() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM camera_events WHERE event_id = %s", (result_id,))
            self.assertEqual(cur.fetchone()[0], 1)


if __name__ == "__main__":
    unittest.main()
