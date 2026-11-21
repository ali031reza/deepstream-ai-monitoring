"""Verify a session's persisted clean and boxed JPEG references."""
def verify_session(database, store, session_id, expected_events):
    with database.transaction() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, is_generated
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'camera_events'
            """
        )
        columns = {name: generated for name, generated in cur.fetchall()}
        root_sql = "event_storage_root" if "event_storage_root" in columns else "NULL::text"
        if "event_bucket_stored" in columns:
            bucket_sql = "event_bucket_stored"
        elif "event_bucket" in columns and columns["event_bucket"] != "ALWAYS":
            bucket_sql = "event_bucket"
        else:
            bucket_sql = "event_bucket"
        cur.execute(
            f"""
            SELECT model_id, {root_sql}, {bucket_sql}, clean_key, clean_bytes, clean_sha256,
                   boxed_key, boxed_bytes, boxed_sha256
            FROM camera_events WHERE source_session_id = %s
            """,
            (session_id,),
        )
        rows = cur.fetchall()
    models = {}
    verified = set()
    for model, root, bucket, clean, size, digest, boxed, boxed_size, boxed_digest in rows:
        counts = models.setdefault(model, {"rows": 0, "events": 0})
        counts["rows"] += 1
        counts["events"] += 1
        if len(rows) == expected_events:
            legacy = store.boxed_key(clean)
            model_specific = store.boxed_key_for(clean, model)
            if boxed not in {legacy, model_specific}:
                raise AssertionError(f"Invalid boxed object for {clean}")
            if root is not None and root != store.root:
                raise AssertionError(f"Invalid storage root for {clean}")
            if bucket != store.bucket:
                raise AssertionError(f"Invalid event bucket for {clean}")
            for key, length, sha in ((clean, size, digest), (boxed, boxed_size, boxed_digest)):
                store.verify(storage_key=key, image_size_bytes=length, image_sha256=sha)
                verified.add(key)
    return {"source_session_id": session_id, "models": models, "verified_objects": len(verified),
            "verified_unique_images": len(verified) // 2}
