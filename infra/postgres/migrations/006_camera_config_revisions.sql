-- Immutable camera configuration revisions.
-- Safe to re-run. Existing camera_events stay unversioned (config_version NULL).
-- This migration does not copy today's settings onto historical events.

BEGIN;

ALTER TABLE public.cameras
    ADD COLUMN IF NOT EXISTS region text,
    ADD COLUMN IF NOT EXISTS current_config_version integer;

CREATE TABLE IF NOT EXISTS public.camera_config_revisions (
    camera_id text NOT NULL REFERENCES public.cameras (camera_id),
    config_version integer NOT NULL CHECK (config_version > 0),
    zone text NOT NULL,
    region text,
    available_models text[] NOT NULL,
    active_models text[] NOT NULL,
    event_storage_root text NOT NULL CHECK (event_storage_root LIKE 's3://%/%'),
    model_assignments jsonb NOT NULL DEFAULT '[]'::jsonb,
    applied_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (camera_id, config_version),
    CHECK (cardinality(active_models) > 0),
    CHECK (active_models <@ available_models)
);

-- Seed revision 1 from the camera row that already exists. Do not invent a
-- second editable copy: later changes go through tools.apply_camera_config,
-- which updates this pointer and inserts the next revision together.
INSERT INTO public.camera_config_revisions (
    camera_id, config_version, zone, region, available_models, active_models,
    event_storage_root, model_assignments
)
SELECT
    c.camera_id,
    1,
    c.zone,
    c.region,
    c.available_models,
    ARRAY[c.current_model],
    c.event_storage_root,
    '[]'::jsonb
FROM public.cameras AS c
WHERE c.available_models IS NOT NULL
  AND c.current_model IS NOT NULL
  AND c.event_storage_root IS NOT NULL
  AND NOT EXISTS (
      SELECT 1
      FROM public.camera_config_revisions AS r
      WHERE r.camera_id = c.camera_id
  );

-- The checked-in migrations have no camera_model_assignments table. If a
-- deployment already has one, snapshot its rows into revision 1 before any
-- event references that revision. Referenced revisions are never rewritten.
DO $snapshot$
BEGIN
    IF to_regclass('public.camera_model_assignments') IS NOT NULL
       AND EXISTS (
           SELECT 1
           FROM information_schema.columns
           WHERE table_schema = 'public'
             AND table_name = 'camera_model_assignments'
             AND column_name = 'camera_id'
       )
       AND NOT EXISTS (
           SELECT 1 FROM pg_trigger
           WHERE tgname = 'camera_config_revisions_immutable'
             AND tgrelid = 'public.camera_config_revisions'::regclass
             AND NOT tgisinternal
       )
    THEN
        EXECUTE $sql$
            UPDATE public.camera_config_revisions AS r
            SET model_assignments = COALESCE((
                SELECT jsonb_agg(to_jsonb(a) ORDER BY a.ctid)
                FROM public.camera_model_assignments AS a
                WHERE a.camera_id = r.camera_id
            ), '[]'::jsonb)
            WHERE r.model_assignments = '[]'::jsonb
        $sql$;
    END IF;
END
$snapshot$;

UPDATE public.cameras AS c
SET current_config_version = 1
WHERE c.current_config_version IS NULL
  AND EXISTS (
      SELECT 1
      FROM public.camera_config_revisions AS r
      WHERE r.camera_id = c.camera_id
        AND r.config_version = 1
  );

ALTER TABLE public.camera_events
    ADD COLUMN IF NOT EXISTS config_version integer;

DO $fk$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'camera_events_config_revision_fkey'
    ) THEN
        ALTER TABLE public.camera_events
            ADD CONSTRAINT camera_events_config_revision_fkey
            FOREIGN KEY (camera_id, config_version)
            REFERENCES public.camera_config_revisions (camera_id, config_version);
    END IF;
END
$fk$;

CREATE OR REPLACE FUNCTION public.reject_camera_config_revision_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'camera configuration revision %/% is immutable',
        COALESCE(OLD.camera_id, NEW.camera_id),
        COALESCE(OLD.config_version, NEW.config_version);
END;
$$;

DROP TRIGGER IF EXISTS camera_config_revisions_immutable ON public.camera_config_revisions;
CREATE TRIGGER camera_config_revisions_immutable
    BEFORE UPDATE OR DELETE ON public.camera_config_revisions
    FOR EACH ROW
    EXECUTE FUNCTION public.reject_camera_config_revision_mutation();

CREATE OR REPLACE FUNCTION public.guard_camera_config_update()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF current_setting('camera_config.apply', true) IS DISTINCT FROM 'on'
       AND (
           NEW.zone IS DISTINCT FROM OLD.zone
           OR NEW.region IS DISTINCT FROM OLD.region
           OR NEW.available_models IS DISTINCT FROM OLD.available_models
           OR NEW.current_model IS DISTINCT FROM OLD.current_model
           OR NEW.event_storage_root IS DISTINCT FROM OLD.event_storage_root
           OR NEW.current_config_version IS DISTINCT FROM OLD.current_config_version
       )
    THEN
        RAISE EXCEPTION
            'Change camera configuration with tools.apply_camera_config so the revision stays atomic';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS cameras_config_guard ON public.cameras;
CREATE TRIGGER cameras_config_guard
    BEFORE UPDATE ON public.cameras
    FOR EACH ROW
    EXECUTE FUNCTION public.guard_camera_config_update();

DO $grant$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ai_app') THEN
        GRANT SELECT, UPDATE ON public.cameras TO ai_app;
        GRANT SELECT, INSERT ON public.camera_config_revisions TO ai_app;
        GRANT SELECT, INSERT ON public.camera_events TO ai_app;
    END IF;
END
$grant$;

COMMIT;
