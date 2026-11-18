BEGIN;

CREATE TABLE IF NOT EXISTS cameras (
    camera_id text PRIMARY KEY,
    zone text NOT NULL,
    available_models text[] NOT NULL,
    current_model text NOT NULL CHECK (current_model = ANY(available_models)),
    event_storage_root text NOT NULL CHECK (event_storage_root LIKE 's3://%/%')
);

INSERT INTO cameras VALUES
    ('camera-01', 'A', ARRAY['fire', 'person'], 'person', 's3://camera-events/events/camera-01')
ON CONFLICT (camera_id) DO NOTHING;

CREATE TABLE IF NOT EXISTS camera_events (
    event_id text PRIMARY KEY,
    camera_id text NOT NULL REFERENCES cameras(camera_id),
    source_session_id text NOT NULL,
    frame_sequence bigint NOT NULL,
    model_id text NOT NULL,
    event_type text NOT NULL,
    event_at timestamptz NOT NULL,
    source_video_ms double precision NOT NULL,
    event_storage_root text NOT NULL,
    clean_key text NOT NULL,
    clean_bytes bigint NOT NULL CHECK (clean_bytes > 0),
    clean_sha256 char(64) NOT NULL,
    boxed_key text NOT NULL,
    boxed_bytes bigint NOT NULL CHECK (boxed_bytes > 0),
    boxed_sha256 char(64) NOT NULL,
    detections jsonb NOT NULL,
    payload_sha256 char(64) NOT NULL,
    UNIQUE (source_session_id, model_id, frame_sequence)
);

CREATE INDEX IF NOT EXISTS camera_events_camera_time ON camera_events (camera_id, event_at DESC);
COMMIT;
