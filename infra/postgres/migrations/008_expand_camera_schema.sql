-- Expand the camera schema. Safe to re-run. Does not drop columns.
--
-- Authoritative model state moves to camera_config_revision_models, one row
-- per model available on that revision. is_active marks the enabled ones.
-- Historical revisions are backfilled from their own available_models and
-- active_models arrays, not from the camera's current settings.
--
-- model_assignments JSON is left in place. A nonempty JSON snapshot is
-- reported rather than rewritten. The immutability trigger already forbids
-- changing a revision row.
--
-- The live database also has camera_model_assignments (camera_id, model_id,
-- enabled, assigned_at), keyed only by camera and model and cascading from
-- cameras and models. It records the current registration for the older
-- camera ids. It has no config_version, so a new revision would overwrite
-- history, and its model ids (person_face, fire, ppe, car) are not the
-- pipeline assignment ids. This migration leaves that table untouched and
-- stores revision history in camera_config_revision_models.
--
-- camera_events.event_bucket stays generated until 009. event_bucket_stored
-- is an independent copy, verified against the generated bucket before any
-- later drop of event_storage_root. primary_detection_index is a zero-based
-- index into detections. It is filled from that array, never by parsing the
-- legacy person_id text. Unknown historical primaries and engine builds stay
-- NULL.

BEGIN;

ALTER TABLE public.cameras
    ADD COLUMN IF NOT EXISTS rtsp_url text,
    ADD COLUMN IF NOT EXISTS credential_ref text,
    ADD COLUMN IF NOT EXISTS enabled boolean NOT NULL DEFAULT true,
    ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now();

CREATE TABLE IF NOT EXISTS public.camera_config_revision_models (
    camera_id text NOT NULL,
    config_version integer NOT NULL,
    model_id text NOT NULL,
    is_active boolean NOT NULL,
    PRIMARY KEY (camera_id, config_version, model_id),
    FOREIGN KEY (camera_id, config_version)
        REFERENCES public.camera_config_revisions (camera_id, config_version),
    CHECK (length(btrim(model_id)) > 0)
);

INSERT INTO public.camera_config_revision_models (camera_id, config_version, model_id, is_active)
SELECT r.camera_id, r.config_version, model.model_id, model.model_id = ANY (r.active_models)
FROM public.camera_config_revisions AS r
CROSS JOIN LATERAL unnest(r.available_models) AS model(model_id)
ON CONFLICT (camera_id, config_version, model_id) DO NOTHING;

-- Keep assignment rows immutable once an event can point at the revision.
CREATE OR REPLACE FUNCTION public.reject_camera_config_model_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'camera configuration assignment %/%/% is immutable',
        COALESCE(OLD.camera_id, NEW.camera_id),
        COALESCE(OLD.config_version, NEW.config_version),
        COALESCE(OLD.model_id, NEW.model_id);
END;
$$;

DROP TRIGGER IF EXISTS camera_config_revision_models_immutable
    ON public.camera_config_revision_models;
CREATE TRIGGER camera_config_revision_models_immutable
    BEFORE UPDATE OR DELETE ON public.camera_config_revision_models
    FOR EACH ROW
    EXECUTE FUNCTION public.reject_camera_config_model_mutation();

DO $pointer$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'cameras_current_revision_fkey'
    ) THEN
        ALTER TABLE public.cameras
            ADD CONSTRAINT cameras_current_revision_fkey
            FOREIGN KEY (camera_id, current_config_version)
            REFERENCES public.camera_config_revisions (camera_id, config_version);
    END IF;
END
$pointer$;

ALTER TABLE public.camera_events
    ADD COLUMN IF NOT EXISTS event_bucket_stored text,
    ADD COLUMN IF NOT EXISTS primary_class text,
    ADD COLUMN IF NOT EXISTS primary_confidence double precision,
    ADD COLUMN IF NOT EXISTS primary_detection_index integer,
    ADD COLUMN IF NOT EXISTS engine_digest text;

DO $confidence$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'camera_events_primary_confidence_range'
    ) THEN
        ALTER TABLE public.camera_events
            ADD CONSTRAINT camera_events_primary_confidence_range
            CHECK (
                primary_confidence IS NULL
                OR (primary_confidence >= 0 AND primary_confidence <= 1)
            );
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'camera_events_primary_index_nonnegative'
    ) THEN
        ALTER TABLE public.camera_events
            ADD CONSTRAINT camera_events_primary_index_nonnegative
            CHECK (primary_detection_index IS NULL OR primary_detection_index >= 0);
    END IF;
END
$confidence$;

UPDATE public.camera_events
SET event_bucket_stored = event_bucket
WHERE event_bucket_stored IS NULL
  AND event_bucket IS NOT NULL;

DO $bucket_check$
DECLARE
    mismatched bigint;
BEGIN
    SELECT count(*) INTO mismatched
    FROM public.camera_events
    WHERE event_bucket IS DISTINCT FROM event_bucket_stored
       OR (
           event_storage_root IS NOT NULL
           AND split_part(event_storage_root, '/', 3) IS DISTINCT FROM event_bucket_stored
       );
    IF mismatched > 0 THEN
        RAISE EXCEPTION
            'event bucket backfill mismatch on % camera_events rows', mismatched;
    END IF;
END
$bucket_check$;

-- Highest confidence, then the earliest array element. Ordinality is 1-based,
-- so the stored index is ord - 1. Index 0 is a real detection.
WITH picked AS (
    SELECT event_id, class_name, confidence, idx
    FROM (
        SELECT
            e.event_id,
            det->>'class_name' AS class_name,
            (det->>'confidence')::double precision AS confidence,
            (item.ord - 1)::integer AS idx,
            row_number() OVER (
                PARTITION BY e.event_id
                ORDER BY (det->>'confidence')::double precision DESC, item.ord ASC
            ) AS rank
        FROM public.camera_events AS e
        CROSS JOIN LATERAL jsonb_array_elements(e.detections)
            WITH ORDINALITY AS item(det, ord)
        WHERE jsonb_typeof(det->'confidence') = 'number'
          AND lower(det->>'class_name') = 'person'
          AND (det->>'confidence')::double precision >= 0.65
    ) AS ranked
    WHERE rank = 1
)
UPDATE public.camera_events AS e
SET primary_class = picked.class_name,
    primary_confidence = picked.confidence,
    primary_detection_index = picked.idx
FROM picked
WHERE e.event_id = picked.event_id
  AND e.primary_detection_index IS NULL
  AND e.primary_class IS NULL
  AND e.primary_confidence IS NULL;

CREATE OR REPLACE VIEW public.camera_current_config AS
SELECT
    c.camera_id,
    c.name,
    c.rtsp_url,
    c.credential_ref,
    c.enabled,
    c.created_at,
    c.current_config_version,
    r.zone,
    r.region,
    r.event_storage_root,
    split_part(r.event_storage_root, '/', 3) AS event_bucket,
    COALESCE(available.models, ARRAY[]::text[]) AS available_models,
    COALESCE(active.models, ARRAY[]::text[]) AS active_models
FROM public.cameras AS c
JOIN public.camera_config_revisions AS r
  ON r.camera_id = c.camera_id
 AND r.config_version = c.current_config_version
LEFT JOIN LATERAL (
    SELECT array_agg(m.model_id ORDER BY m.model_id) AS models
    FROM public.camera_config_revision_models AS m
    WHERE m.camera_id = r.camera_id
      AND m.config_version = r.config_version
) AS available ON true
LEFT JOIN LATERAL (
    SELECT array_agg(m.model_id ORDER BY m.model_id) AS models
    FROM public.camera_config_revision_models AS m
    WHERE m.camera_id = r.camera_id
      AND m.config_version = r.config_version
      AND m.is_active
) AS active ON true;

COMMENT ON VIEW public.camera_current_config IS
    'Registration fields plus the current revision. event_bucket is the bucket name in event_storage_root. available_models and active_models come from camera_config_revision_models. credential_ref is a reference, not the secret. Grant matches cameras: the application role can read it.';

DO $notice$
DECLARE
    nonempty bigint;
    extra_keys text;
BEGIN
    SELECT count(*) INTO nonempty
    FROM public.camera_config_revisions
    WHERE model_assignments <> '[]'::jsonb;
    IF nonempty = 0 THEN
        RAISE NOTICE 'camera_config_revisions.model_assignments has no nonempty snapshots; nothing unmapped';
    ELSE
        SELECT string_agg(DISTINCT key, ', ' ORDER BY key) INTO extra_keys
        FROM public.camera_config_revisions AS r
        CROSS JOIN LATERAL jsonb_array_elements(r.model_assignments) AS elem
        CROSS JOIN LATERAL jsonb_object_keys(elem) AS key
        WHERE jsonb_typeof(elem) = 'object'
          AND key NOT IN ('model_id', 'model', 'is_active', 'active');
        RAISE NOTICE
            'retained % nonempty model_assignments snapshots; unmapped keys: %',
            nonempty, COALESCE(extra_keys, '(none beyond model identity)');
    END IF;
END
$notice$;

DO $grant$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ai_app') THEN
        GRANT SELECT, INSERT ON public.camera_config_revision_models TO ai_app;
        GRANT SELECT ON public.camera_current_config TO ai_app;
        GRANT SELECT, INSERT ON public.camera_events TO ai_app;
        GRANT SELECT, UPDATE ON public.cameras TO ai_app;
    END IF;
END
$grant$;

COMMIT;
