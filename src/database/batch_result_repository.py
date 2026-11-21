"""Idempotent persistence of timestamped events and immutable JPEG references."""
from datetime import datetime, timezone
import hashlib

from psycopg.types.json import Jsonb

from src.database.camera_config import event_config_version


def _blank_to_none(value: str) -> str | None:
    text = value.strip()
    return text or None


class DatabaseBatchRepository:
    def __init__(self, database):
        self.database = database

    def persist_batch(self, results):
        rows = {}
        for result in results:
            if not result.is_event:
                continue
            digest = hashlib.sha256(result.SerializeToString(deterministic=True)).hexdigest()
            if result.result_id in rows and rows[result.result_id][-1] != digest:
                raise ValueError(f"Conflicting event {result.result_id}")
            detections = []
            for detection in result.detections:
                item = {"class_id": detection.class_id, "class_name": detection.class_name,
                        "confidence": detection.confidence,
                        "bbox_normalized": list(detection.bbox_normalized)}
                if detection.person_id:
                    item["person_id"] = detection.person_id
                detections.append(item)
            rows[result.result_id] = (
                result.result_id, result.camera_id, result.source_session_id, result.frame_sequence,
                result.model_id, result.event_type,
                datetime.fromtimestamp(result.captured_at_ms / 1000, tz=timezone.utc),
                result.source_video_ms, result.event_storage_root,
                result.storage_key, result.image_size_bytes, result.image_sha256,
                result.boxed_key, result.boxed_bytes, result.boxed_sha256, Jsonb(detections),
                _blank_to_none(result.event),
                None if result.confidence == 0 and not result.event.strip() else result.confidence,
                _blank_to_none(result.person_id),
                event_config_version(result.camera_config_version), digest,
            )
        if not rows:
            return
        with self.database.transaction() as conn, conn.cursor() as cur:
            cur.executemany("""
                INSERT INTO camera_events (
                    event_id, camera_id, source_session_id, frame_sequence, model_id, event_type,
                    event_at, source_video_ms, event_storage_root, clean_key, clean_bytes, clean_sha256,
                    boxed_key, boxed_bytes, boxed_sha256, detections, event, confidence, person_id,
                    config_version, payload_sha256
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (event_id) DO NOTHING
            """, rows.values())
            cur.execute("SELECT event_id, payload_sha256 FROM camera_events WHERE event_id = ANY(%s)",
                        (list(rows),))
            stored = dict(cur.fetchall())
            if any(stored.get(key) != row[-1] for key, row in rows.items()):
                raise ValueError("Event replay conflicts with stored payload")
