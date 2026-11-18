BEGIN;

-- Object keys are relative to the bucket. Keep its explicit name in sync
-- with the S3 root selected for the camera or recorded with the event.
ALTER TABLE public.cameras
    ADD COLUMN IF NOT EXISTS event_bucket text
    GENERATED ALWAYS AS (split_part(event_storage_root, '/', 3)) STORED;

ALTER TABLE public.camera_events
    ADD COLUMN IF NOT EXISTS event_bucket text
    GENERATED ALWAYS AS (split_part(event_storage_root, '/', 3)) STORED;

COMMIT;
