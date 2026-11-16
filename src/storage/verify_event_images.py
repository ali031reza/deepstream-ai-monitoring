"""Read-only PostgreSQL metadata ↔ persistent JPEG integrity verification."""
from __future__ import annotations
from configs.settings import load_settings
import argparse
import json

from src.storage.event_frame_store import EventFrameStore


def verify_session(database, store: EventFrameStore, session_id: str):
    """Return counts; raise if any expected image metadata/file is inconsistent."""
    with database.transaction() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT model_id, COUNT(*), COUNT(*) FILTER (WHERE is_event),
                       COUNT(*) FILTER (WHERE is_event AND event_image_id IS NULL)
                FROM inference_results
                WHERE source_session_id = %s
                GROUP BY model_id
                ORDER BY model_id
            """, (session_id,))
            model_rows = cur.fetchall()
            cur.execute("""
                SELECT DISTINCT e.storage_key, e.file_size_bytes,
                       e.sha256, e.image_format, e.camera_id, e.frame_id
                FROM event_images e
                JOIN inference_results r ON r.event_image_id = e.id
                WHERE r.source_session_id = %s
                ORDER BY e.storage_key
            """, (session_id,))
            image_rows = cur.fetchall()
    models = {}
    for model_id, count, events, missing in model_rows:
        if missing:
            raise AssertionError(f"{model_id} has {missing} event results without image ID")
        models[model_id] = {"rows": count, "events": events}
    for key, size, sha, fmt, camera, frame_id in image_rows:
        if fmt.lower() not in ("jpg", "jpeg"):
            raise AssertionError(f"Unexpected image format {fmt} for {key}")
        store.verify(storage_key=key, image_size_bytes=size, image_sha256=sha.strip())
    return {
        "source_session_id": session_id,
        "models": models, "verified_unique_images": len(image_rows),
    }


def main():
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--event-root", default=str(settings.events.root))
    parser.add_argument("--kafka-config", default=str(settings.paths.config_file))
    args = parser.parse_args()
    from src.kafka.runtime_config import load_runtime_config
    from src.database.client import DatabaseClient
    database = DatabaseClient(**load_runtime_config(args.kafka_config)["database"])
    result = verify_session(database, EventFrameStore(args.event_root), args.session_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["models"]:
        raise SystemExit("No rows found for this source session")
    print("PASS: PostgreSQL event JPEG references verified")


if __name__ == "__main__":
    main()
