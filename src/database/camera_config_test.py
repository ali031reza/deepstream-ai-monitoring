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
        self.assertEqual(fields[30], "event")
        self.assertEqual(fields[31], "confidence")
        self.assertEqual(fields[32], "person_id")
        self.assertEqual(fields[33], "primary_detection_index")
        self.assertEqual(fields[34], "primary_class")
        self.assertEqual(fields[35], "primary_confidence")
        self.assertEqual(fields[36], "engine_digest")
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

    def test_primary_detection_keeps_array_order_and_index_zero(self):
        from src.pipeline.run_tensorrt_event_pipeline import select_primary_detection

        tied = [
            {"class_name": "person", "confidence": 0.80},
            {"class_name": "person", "confidence": 0.80},
        ]
        self.assertEqual(
            select_primary_detection(tied, detection_class="person", minimum_confidence=0.65),
            0,
        )
        unsorted = [
            {"class_name": "person", "confidence": 0.70},
            {"class_name": "fire", "confidence": 0.99},
            {"class_name": "person", "confidence": 0.91},
        ]
        self.assertEqual(
            select_primary_detection(unsorted, detection_class="person", minimum_confidence=0.65),
            2,
        )
        self.assertEqual(unsorted[0]["confidence"], 0.70)

    def test_assignment_order_is_not_a_configuration_change(self):
        from src.database.camera_config import assignment_identity

        self.assertEqual(
            assignment_identity(("fire", "person"), ("person",)),
            assignment_identity(("person", "fire"), ("person",)),
        )
        self.assertNotEqual(
            assignment_identity(("fire", "person"), ("person",)),
            assignment_identity(("fire", "person"), ("person", "fire")),
        )

    def test_legacy_person_id_is_not_used_as_an_index(self):
        from src.database.batch_result_repository import _primary_index
        from src.proto.inference_result_pb2 import InferenceResult

        legacy = InferenceResult(event="person", confidence=0.8, person_id="6090:1")
        self.assertFalse(legacy.HasField("primary_detection_index"))
        self.assertIsNone(_primary_index(legacy))
        current = InferenceResult(primary_class="person", primary_confidence=0.8)
        current.primary_detection_index = 0
        self.assertEqual(_primary_index(current), 0)

    def test_kafka_event_carries_primary_detection(self):
        from src.pipeline.run_tensorrt_event_pipeline import select_primary_detection
        from src.results.protobuf_result_adapter import build_inference_result

        detections = [
            {"class_id": 0, "class_name": "person", "confidence": 0.70,
             "bbox_normalized": [0.5, 0.2, 0.8, 0.9], "synthetic": False},
            {"class_id": 0, "class_name": "person", "confidence": 0.91,
             "bbox_normalized": [0.1, 0.1, 0.4, 0.8], "synthetic": False},
        ]
        primary_index = select_primary_detection(
            detections, detection_class="person", minimum_confidence=0.65
        )
        message = build_inference_result(
            {
                "model_id": "yolo11n-person",
                "frame_id": 6090,
                "input_frame_shape": [1080, 1920, 3],
                "status": "processed",
                "is_event": True,
                "event_type": "person_detected",
                "storage_key": "events/camera-01/session/000000006090.jpg",
                "event_storage_root": "s3://camera-events/events/camera-01",
                "boxed_key": "events/camera-01/session/000000006090.boxed.jpg",
                "boxed_sha256": "b" * 64,
                "boxed_bytes": 11,
                "source_video_ms": 1000.0,
                "image_format": "jpeg",
                "image_size_bytes": 10,
                "image_sha256": "a" * 64,
                "primary_class": detections[primary_index]["class_name"],
                "primary_confidence": detections[primary_index]["confidence"],
                "primary_detection_index": primary_index,
                "engine_digest": "c" * 64,
                "detections": detections,
            },
            camera_id="camera-01",
            source_session_id="session",
            frame_sequence=6090,
            captured_at_ms=1,
            worker_id="worker",
            camera_config_version=1,
        )
        self.assertEqual(message.primary_class, "person")
        self.assertAlmostEqual(message.primary_confidence, 0.91, places=5)
        self.assertEqual(message.primary_detection_index, 1)
        self.assertTrue(message.HasField("primary_detection_index"))
        self.assertEqual(message.engine_digest, "c" * 64)
        self.assertEqual(message.schema_version, "2.0")
        self.assertFalse(message.person_id)


class CameraConfigRevisionTest(unittest.TestCase):
    def setUp(self):
        try:
            self.database = DatabaseClient(**load_settings().database.connection_kwargs())
            self.database.check_connection()
        except Exception as exc:
            self.skipTest(f"PostgreSQL is unavailable: {exc}")
        with self.database.transaction() as conn, conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.camera_config_revision_models')")
            if cur.fetchone()[0] is None:
                self.skipTest("Apply infra/postgres/migrations/008_expand_camera_schema.sql first")

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
                            SELECT column_name FROM information_schema.columns
                            WHERE table_schema = 'public' AND table_name = 'cameras'
                            """
                        )
                        camera_columns = {row[0] for row in cur.fetchall()}
                        if "zone" in camera_columns:
                            cur.execute(
                                """
                                INSERT INTO public.cameras (
                                    camera_id, name, zone, available_models, current_model, event_storage_root
                                ) VALUES (%s, %s, 'A', ARRAY['fire','person'], 'person', %s)
                                """,
                                (fixture.camera_id, fixture.camera_id, fixture.event_storage_root),
                            )
                        else:
                            cur.execute(
                                "INSERT INTO public.cameras (camera_id, name) VALUES (%s, %s)",
                                (fixture.camera_id, fixture.camera_id),
                            )
                    first = apply_camera_config(conn, fixture)
                    second = apply_camera_config(conn, fixture)
                    self.assertEqual(first, ApplyResult(fixture.camera_id, 1, True))
                    self.assertEqual(second, ApplyResult(fixture.camera_id, 1, False))
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT column_name, is_generated
                            FROM information_schema.columns
                            WHERE table_schema = 'public' AND table_name = 'camera_events'
                            """
                        )
                        event_columns = {name: generated for name, generated in cur.fetchall()}
                        names = [
                            "event_id", "camera_id", "source_session_id", "frame_sequence", "model_id",
                            "event_type", "event_at", "source_video_ms", "event_storage_root",
                            "event_bucket_stored", "event_bucket",
                            "clean_key", "clean_bytes", "clean_sha256", "boxed_key", "boxed_bytes",
                            "boxed_sha256", "detections", "payload_sha256", "config_version",
                        ]
                        values = {
                            "event_id": "fixture-event",
                            "camera_id": fixture.camera_id,
                            "source_session_id": "fixture-session",
                            "frame_sequence": 1,
                            "model_id": "yolo11n-person",
                            "event_type": "person_detected",
                            "event_at": "now()",
                            "source_video_ms": 0,
                            "event_storage_root": fixture.event_storage_root,
                            "event_bucket_stored": "camera-events",
                            "event_bucket": "camera-events",
                            "clean_key": "events/camera-config-fixture/fixture-session/000000000001.jpg",
                            "clean_bytes": 10,
                            "clean_sha256": "a" * 64,
                            "boxed_key": "events/camera-config-fixture/fixture-session/000000000001.boxed.jpg",
                            "boxed_bytes": 11,
                            "boxed_sha256": "b" * 64,
                            "detections": "[]",
                            "payload_sha256": "c" * 64,
                            "config_version": 1,
                        }
                        chosen = [
                            name for name in names
                            if name in event_columns and event_columns[name] != "ALWAYS"
                        ]
                        placeholders = []
                        parameters = []
                        for name in chosen:
                            if name in {"event_at", "detections"}:
                                placeholders.append("now()" if name == "event_at" else "%s::jsonb")
                            else:
                                placeholders.append("%s")
                            if name != "event_at":
                                parameters.append(values[name])
                        cur.execute(
                            f"INSERT INTO public.camera_events ({', '.join(chosen)}) "
                            f"VALUES ({', '.join(placeholders)})",
                            parameters,
                        )
                    third = apply_camera_config(conn, changed)
                    self.assertEqual(third, ApplyResult(fixture.camera_id, 2, True))
                    fourth = apply_camera_config(conn, changed)
                    self.assertEqual(fourth, ApplyResult(fixture.camera_id, 2, False))
                    reordered = replace(changed, available_models=("person", "fire"))
                    fifth = apply_camera_config(conn, reordered)
                    self.assertEqual(fifth, ApplyResult(fixture.camera_id, 2, False))
                    activated = replace(changed, active_models=("fire", "person"))
                    sixth = apply_camera_config(conn, activated)
                    self.assertEqual(sixth, ApplyResult(fixture.camera_id, 3, True))
                    with conn.cursor() as cur:
                        cur.execute(
                            """
                            SELECT r.zone, current_config.zone, current_config.current_config_version,
                                   array_agg(m.model_id ORDER BY m.model_id) FILTER (WHERE m.is_active),
                                   array_agg(m.model_id ORDER BY m.model_id) FILTER (WHERE NOT m.is_active)
                            FROM public.camera_events AS e
                            JOIN public.camera_config_revisions AS r
                              ON r.camera_id = e.camera_id
                             AND r.config_version = e.config_version
                            JOIN public.camera_current_config AS current_config
                              ON current_config.camera_id = e.camera_id
                            JOIN public.camera_config_revision_models AS m
                              ON m.camera_id = r.camera_id
                             AND m.config_version = r.config_version
                            WHERE e.event_id = 'fixture-event'
                            GROUP BY r.zone, current_config.zone, current_config.current_config_version
                            """
                        )
                        historical_zone, current_zone, current_version, historical_active, historical_inactive = cur.fetchone()
                    self.assertEqual(historical_zone, "A")
                    self.assertEqual(current_zone, "B")
                    self.assertEqual(current_version, 3)
                    self.assertEqual(list(historical_active), ["person"])
                    self.assertEqual(list(historical_inactive), ["fire"])
                    raise _Rollback()
            except _Rollback:
                pass


class _Rollback(Exception):
    pass


if __name__ == "__main__":
    unittest.main()
