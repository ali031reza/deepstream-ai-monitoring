"""Camera configuration revisions stay attached to the events that used them."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

import psycopg

from configs.settings import load_settings
from src.database.camera_config import (
    ApplyResult,
    apply_camera_config,
    event_config_version,
    load_camera_config_file,
)
from src.database.client import DatabaseClient


ROOT = Path(__file__).resolve().parents[2]
CAMERA_FILE = ROOT / "infra" / "postgres" / "cameras_counfig" / "camera-01.yaml"


class CameraConfigFileTest(unittest.TestCase):
    def test_camera_01_file_has_no_credentials_and_matches_the_registry(self):
        loaded = load_camera_config_file(CAMERA_FILE)
        self.assertEqual(loaded.camera_id, "camera-01")
        self.assertEqual(loaded.zone, "A")
        self.assertIsNone(loaded.region)
        self.assertEqual(loaded.available_models, ("fire", "person"))
        self.assertEqual(loaded.active_models, ("person",))
        self.assertEqual(loaded.event_storage_root, "s3://camera-events/events/camera-01")

    def test_credential_fields_are_rejected(self):
        secret = ROOT / "infra" / "postgres" / "cameras_counfig" / "_secret_test.yaml"
        secret.write_text("camera_id: camera-01\npassword: hidden\n", encoding="utf-8")
        try:
            with self.assertRaises(ValueError):
                load_camera_config_file(secret)
        finally:
            secret.unlink()

    def test_missing_protobuf_revision_stays_unversioned(self):
        from src.proto.inference_result_pb2 import InferenceResult

        self.assertIsNone(event_config_version(0))
        self.assertEqual(event_config_version(2), 2)
        fields = {field.number: field.name for field in InferenceResult.DESCRIPTOR.fields}
        self.assertEqual(fields[1], "camera_id")
        self.assertEqual(fields[23], "schema_version")
        self.assertEqual(fields[28], "source_video_ms")
        self.assertEqual(fields[29], "camera_config_version")
        message = InferenceResult(schema_version="2.0", camera_config_version=4)
        encoded = message.SerializeToString()
        parsed = InferenceResult()
        parsed.ParseFromString(encoded)
        self.assertEqual(parsed.schema_version, "2.0")
        self.assertEqual(parsed.camera_config_version, 4)
        message.ClearField("camera_config_version")
        older = InferenceResult()
        older.ParseFromString(message.SerializeToString())
        self.assertEqual(older.camera_config_version, 0)
        self.assertIsNone(event_config_version(older.camera_config_version))


class CameraConfigRevisionTest(unittest.TestCase):
    def setUp(self):
        try:
            self.database = DatabaseClient(**load_settings().database.connection_kwargs())
            self.database.check_connection()
        except Exception as exc:
            self.skipTest(f"PostgreSQL is unavailable: {exc}")
        with self.database.transaction() as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.camera_config_revisions')")
            if cur.fetchone()[0] is None:
                self.skipTest("Apply infra/postgres/migrations/006_camera_config_revisions.sql first")

    def test_zone_change_leaves_the_earlier_event_on_zone_a(self):
        desired = load_camera_config_file(CAMERA_FILE)
        fixture = desired.__class__(
            camera_id="camera-config-fixture",
            zone="A",
            region=None,
            available_models=("fire", "person"),
            active_models=("person",),
            event_storage_root="s3://camera-events/events/camera-config-fixture",
        )
        changed = replace(fixture, zone="B")
        kwargs = self.database._config
        with psycopg.connect(**kwargs) as conn:
            try:
                with conn.transaction():
                    with conn.cursor() as cur:
                        cur.execute("SELECT set_config('camera_config.apply', 'on', true)")
                        cur.execute(
                            """
                            INSERT INTO public.cameras (
                                camera_id, name, zone, available_models, current_model, event_storage_root
                            ) VALUES (%s, %s, 'A', ARRAY['fire','person'], 'person', %s)
                            """,
                            (fixture.camera_id, fixture.camera_id, fixture.event_storage_root),
                        )
                    first = apply_camera_config(conn, fixture)
                    second = apply_camera_config(conn, fixture)
                    self.assertEqual(first, ApplyResult(fixture.camera_id, 1, True))
                    self.assertEqual(second, ApplyResult(fixture.camera_id, 1, False))
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            INSERT INTO public.camera_events (
                                event_id, camera_id, source_session_id, frame_sequence, model_id,
                                event_type, event_at, source_video_ms, event_storage_root,
                                clean_key, clean_bytes, clean_sha256, boxed_key, boxed_bytes,
                                boxed_sha256, detections, payload_sha256, config_version
                            ) VALUES (
                                'fixture-event', %s, 'fixture-session', 1, 'yolo11n-person',
                                'person_detected', now(), 0, %s,
                                'events/camera-config-fixture/fixture-session/000000000001.jpg', 10, %s,
                                'events/camera-config-fixture/fixture-session/000000000001.boxed.jpg', 11, %s,
                                '[]'::jsonb, %s, 1
                            )
                            """,
                            (
                                fixture.camera_id,
                                fixture.event_storage_root,
                                "a" * 64,
                                "b" * 64,
                                "c" * 64,
                            ),
                        )
                    third = apply_camera_config(conn, changed)
                    self.assertEqual(third, ApplyResult(fixture.camera_id, 2, True))
                    fourth = apply_camera_config(conn, changed)
                    self.assertEqual(fourth, ApplyResult(fixture.camera_id, 2, False))
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT r.zone, c.zone, c.current_config_version
                            FROM public.camera_events AS e
                            JOIN public.camera_config_revisions AS r
                              ON r.camera_id = e.camera_id
                             AND r.config_version = e.config_version
                            JOIN public.cameras AS c ON c.camera_id = e.camera_id
                            WHERE e.event_id = 'fixture-event'
                            """
                        )
                        historical_zone, current_zone, current_version = cur.fetchone()
                    self.assertEqual(historical_zone, "A")
                    self.assertEqual(current_zone, "B")
                    self.assertEqual(current_version, 2)
                    raise _Rollback()
            except _Rollback:
                pass


class _Rollback(Exception):
    pass


if __name__ == "__main__":
    unittest.main()
