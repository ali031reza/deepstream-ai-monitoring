"""Verify a session's persisted clean and boxed JPEG references."""
def verify_session(database, store, session_id, expected_events):
    with database.transaction() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT model_id, event_storage_root, clean_key, clean_bytes, clean_sha256,
                   boxed_key, boxed_bytes, boxed_sha256
            FROM camera_events WHERE source_session_id = %s
        """, (session_id,))
        rows = cur.fetchall()
    models = {}
    verified = set()
    for model, root, clean, size, digest, boxed, boxed_size, boxed_digest in rows:
        counts = models.setdefault(model, {"rows": 0, "events": 0})
        counts["rows"] += 1
        counts["events"] += 1
        if len(rows) == expected_events:
            if root != store.root or boxed != store.boxed_key(clean):
                raise AssertionError(f"Invalid object references for {clean}")
            for key, length, sha in ((clean, size, digest), (boxed, boxed_size, boxed_digest)):
                store.verify(storage_key=key, image_size_bytes=length, image_sha256=sha)
                verified.add(key)
    return {"source_session_id": session_id, "models": models, "verified_objects": len(verified),
            "verified_unique_images": len(verified) // 2}
