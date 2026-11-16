-- DeepStream metadata extension for the existing inference_results table.
-- Requires the original schema and event_images migration to be applied.

ALTER TABLE inference_results
    ADD COLUMN IF NOT EXISTS source_session_id TEXT;

ALTER TABLE inference_results
    ADD COLUMN IF NOT EXISTS result_id TEXT;

ALTER TABLE inference_results
    ADD COLUMN IF NOT EXISTS worker_id TEXT;

ALTER TABLE inference_results
    ADD COLUMN IF NOT EXISTS frame_sequence BIGINT;

ALTER TABLE inference_results
    ADD COLUMN IF NOT EXISTS gpu_id INTEGER;

ALTER TABLE inference_results
    ADD COLUMN IF NOT EXISTS frame_width INTEGER;

ALTER TABLE inference_results
    ADD COLUMN IF NOT EXISTS frame_height INTEGER;

ALTER TABLE inference_results
    ADD COLUMN IF NOT EXISTS model_input_size INTEGER;

ALTER TABLE inference_results
    ADD COLUMN IF NOT EXISTS captured_at_ms BIGINT;

ALTER TABLE inference_results
    ADD COLUMN IF NOT EXISTS schema_version TEXT;

ALTER TABLE inference_results
    ADD COLUMN IF NOT EXISTS payload_sha256 CHAR(64);

ALTER TABLE inference_results
    ADD COLUMN IF NOT EXISTS detections JSONB
    NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE inference_results
    ADD COLUMN IF NOT EXISTS timings JSONB
    NOT NULL DEFAULT '[]'::jsonb;

CREATE UNIQUE INDEX IF NOT EXISTS
    uq_inference_results_result_id
    ON inference_results(result_id)
    WHERE result_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS
    idx_inference_results_camera_frame
    ON inference_results(camera_id, frame_id);

CREATE INDEX IF NOT EXISTS
    idx_inference_results_session
    ON inference_results(source_session_id);

CREATE INDEX IF NOT EXISTS
    idx_inference_results_detections
    ON inference_results
    USING GIN(detections);
