BEGIN;

CREATE TABLE IF NOT EXISTS public.cameras (
    camera_id text PRIMARY KEY,
    name text NOT NULL
);

-- Older standalone installations of this migration did not have a name column.
ALTER TABLE public.cameras
    ADD COLUMN IF NOT EXISTS name text,
    ADD COLUMN IF NOT EXISTS zone text NOT NULL DEFAULT 'A',
    ADD COLUMN IF NOT EXISTS available_models text[],
    ADD COLUMN IF NOT EXISTS current_model text CHECK (current_model = ANY(available_models)),
    ADD COLUMN IF NOT EXISTS event_storage_root text CHECK (event_storage_root LIKE 's3://%/%');

INSERT INTO public.cameras AS c
    (camera_id, name, zone, available_models, current_model, event_storage_root)
VALUES
    ('camera-01', 'camera-01', 'A', ARRAY['fire', 'person'], 'person', 's3://camera-events/events/camera-01')
ON CONFLICT (camera_id) DO UPDATE SET
    name = COALESCE(c.name, EXCLUDED.name),
    zone = COALESCE(c.zone, EXCLUDED.zone),
    available_models = COALESCE(c.available_models, EXCLUDED.available_models),
    current_model = COALESCE(c.current_model, EXCLUDED.current_model),
    event_storage_root = COALESCE(c.event_storage_root, EXCLUDED.event_storage_root)
WHERE c.name IS NULL OR c.zone IS NULL OR c.available_models IS NULL
   OR c.current_model IS NULL OR c.event_storage_root IS NULL;

CREATE TABLE IF NOT EXISTS public.camera_events (
    event_id text PRIMARY KEY,
    camera_id text NOT NULL REFERENCES public.cameras(camera_id),
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

CREATE INDEX IF NOT EXISTS camera_events_camera_time ON public.camera_events (camera_id, event_at DESC);
COMMIT;
