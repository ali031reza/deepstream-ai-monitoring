-- Contract migration. Do not apply this to the live ai_monitoring database
-- until every reader of the dropped columns has been checked.
--
-- Prerequisite: application code reads camera_current_config,
-- camera_config_revision_models, event_bucket_stored (or event_bucket after
-- this rename), primary_class, primary_confidence, primary_detection_index,
-- and engine_digest. External UI queries against cameras.zone,
-- cameras.available_models, cameras.current_model, cameras.event_storage_root,
-- camera_config_revisions.available_models, active_models, model_assignments,
-- and camera_events.event, confidence, person_id, or event_storage_root must
-- be moved first. This repository's pipeline is switched. UI consumers
-- outside this repository were not visible from the checkout.
--
-- Safe to re-run only before the drops succeed. A second run after a
-- successful contract fails on missing columns; that is expected.

BEGIN;

DO $ready$
DECLARE
    mismatched bigint;
BEGIN
    SELECT count(*) INTO mismatched
    FROM public.camera_events
    WHERE event_bucket_stored IS NULL
       OR event_bucket IS DISTINCT FROM event_bucket_stored
       OR split_part(event_storage_root, '/', 3) IS DISTINCT FROM event_bucket_stored;
    IF mismatched > 0 THEN
        RAISE EXCEPTION
            'refuse to drop event_storage_root; % events have an unverified bucket',
            mismatched;
    END IF;
END
$ready$;

CREATE OR REPLACE FUNCTION public.guard_camera_config_update()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF current_setting('camera_config.apply', true) IS DISTINCT FROM 'on'
       AND NEW.current_config_version IS DISTINCT FROM OLD.current_config_version
    THEN
        RAISE EXCEPTION
            'Change camera configuration with tools.apply_camera_config so the revision stays atomic';
    END IF;
    RETURN NEW;
END;
$$;

ALTER TABLE public.camera_events DROP COLUMN event_bucket;
ALTER TABLE public.camera_events RENAME COLUMN event_bucket_stored TO event_bucket;
ALTER TABLE public.camera_events ALTER COLUMN event_bucket SET NOT NULL;
ALTER TABLE public.camera_events DROP COLUMN event_storage_root;
ALTER TABLE public.camera_events DROP COLUMN event;
ALTER TABLE public.camera_events DROP COLUMN confidence;
ALTER TABLE public.camera_events DROP COLUMN person_id;

ALTER TABLE public.cameras DROP COLUMN event_bucket;
ALTER TABLE public.cameras DROP COLUMN event_storage_root;
ALTER TABLE public.cameras DROP COLUMN current_model;
ALTER TABLE public.cameras DROP COLUMN available_models;
ALTER TABLE public.cameras DROP COLUMN region;
ALTER TABLE public.cameras DROP COLUMN zone;

ALTER TABLE public.camera_config_revisions DROP COLUMN model_assignments;
ALTER TABLE public.camera_config_revisions DROP COLUMN active_models;
ALTER TABLE public.camera_config_revisions DROP COLUMN available_models;

COMMIT;
