"""Idempotent persistence of timestamped events and immutable JPEG references."""
from datetime import datetime, timezone
import hashlib

from psycopg.types.json import Jsonb

from src.database.camera_config import event_config_version


def _blank_to_none(value: str) -> str | None:
    text = value.strip()
    return text or None


def _bucket(root: str) -> str:
    parts = root.split("/")
    if len(parts) < 3 or not parts[2]:
        raise ValueError("event_storage_root must include a bucket")
    return parts[2]


def _primary_class(result) -> str | None:
    if result.primary_class.strip():
        return result.primary_class.strip()
    return _blank_to_none(result.event)


def _primary_confidence(result) -> float | None:
    if result.primary_class.strip():
        return float(result.primary_confidence)
    if result.event.strip():
        return float(result.confidence)
    return None


def _primary_index(result) -> int | None:
    """Missing stays NULL. Zero is a real detection. Legacy person_id is not an index."""
    if result.HasField("primary_detection_index"):
        return int(result.primary_detection_index)
    return None


def _event_column_specs(cursor) -> list[tuple[str, str | None]]:
    cursor.execute(
        """
        SELECT column_name, is_generated
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'camera_events'
        """
    )
    generated = {name: flag for name, flag in cursor.fetchall()}
    preferred = [
        "event_id", "camera_id", "source_session_id", "frame_sequence", "model_id", "event_type",
        "event_at", "source_video_ms", "event_storage_root", "event_bucket_stored", "event_bucket",
        "clean_key", "clean_bytes", "clean_sha256", "boxed_key", "boxed_bytes", "boxed_sha256",
        "detections", "config_version", "primary_class", "primary_confidence",
        "primary_detection_index", "engine_digest", "payload_sha256",
    ]
    specs = []
    for name in preferred:
        if name not in generated or generated[name] == "ALWAYS":
            continue
        specs.append((name, None))
    if not specs or specs[-1][0] != "payload_sha256":
        raise RuntimeError("camera_events.payload_sha256 is required for replay protection")
    return specs


class DatabaseBatchRepository:
    """Insert events once. A different payload for the same event id is an error.

    event_id is the logical invocation (camera, session, frame, model). Worker
    id and timings are diagnostic fields inside the Protobuf payload. An exact
    retry inserts nothing. A recomputed payload, including one produced by
    another worker or with different timings, keeps the stored row and raises.
    """

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
            rows[result.result_id] = (result, detections, digest)
        if not rows:
            return
        with self.database.transaction() as conn, conn.cursor() as cur:
            specs = _event_column_specs(cur)
            prepared = {}
            for event_id, (result, detections, digest) in rows.items():
                values = {
                    "event_id": result.result_id,
                    "camera_id": result.camera_id,
                    "source_session_id": result.source_session_id,
                    "frame_sequence": result.frame_sequence,
                    "model_id": result.model_id,
                    "event_type": result.event_type,
                    "event_at": datetime.fromtimestamp(result.captured_at_ms / 1000, tz=timezone.utc),
                    "source_video_ms": result.source_video_ms,
                    "event_storage_root": result.event_storage_root,
                    "event_bucket_stored": _bucket(result.event_storage_root),
                    "event_bucket": _bucket(result.event_storage_root),
                    "clean_key": result.storage_key,
                    "clean_bytes": result.image_size_bytes,
                    "clean_sha256": result.image_sha256,
                    "boxed_key": result.boxed_key,
                    "boxed_bytes": result.boxed_bytes,
                    "boxed_sha256": result.boxed_sha256,
                    "detections": Jsonb(detections),
                    "config_version": event_config_version(result.camera_config_version),
                    "primary_class": _primary_class(result),
                    "primary_confidence": _primary_confidence(result),
                    "primary_detection_index": _primary_index(result),
                    "engine_digest": _blank_to_none(result.engine_digest),
                    "payload_sha256": digest,
                }
                prepared[event_id] = tuple(values[name] for name, _cast in specs) 
            names = ", ".join(name for name, _cast in specs)
            placeholders = ", ".join("%s" if cast is None else cast for _name, cast in specs)
            cur.executemany(
                f"INSERT INTO camera_events ({names}) VALUES ({placeholders}) ON CONFLICT (event_id) DO NOTHING",
                prepared.values(),
            )
            cur.execute("SELECT event_id, payload_sha256 FROM camera_events WHERE event_id = ANY(%s)",
                        (list(prepared),))
            stored = dict(cur.fetchall())
            if any(stored.get(key) != row[-1] for key, row in prepared.items()):
                raise ValueError(
                    "Event replay conflicts with stored payload. "
                    "The stored event was kept. A different worker, timing set, or detection "
                    "changes the payload and does not create a second row."
                )
