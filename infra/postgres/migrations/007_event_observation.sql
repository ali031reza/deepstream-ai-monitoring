-- Queryable model observation on each event.
-- Safe to re-run. Existing rows are filled from the detections already stored
-- on the row. person_id is the person number inside that frame: frame_sequence
-- plus ":" plus 1 for the highest-confidence person. The model does not track
-- a person from one frame into the next.

BEGIN;

ALTER TABLE public.camera_events
    ADD COLUMN IF NOT EXISTS event text,
    ADD COLUMN IF NOT EXISTS confidence double precision,
    ADD COLUMN IF NOT EXISTS person_id text;

DO $check$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint WHERE conname = 'camera_events_confidence_range'
    ) THEN
        ALTER TABLE public.camera_events
            ADD CONSTRAINT camera_events_confidence_range
            CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1));
    END IF;
END
$check$;

WITH people AS (
    SELECT
        e.event_id,
        det->>'class_name' AS class_name,
        (det->>'confidence')::double precision AS confidence,
        row_number() OVER (PARTITION BY e.event_id ORDER BY ord) AS person_ordinal
    FROM public.camera_events AS e
    CROSS JOIN LATERAL jsonb_array_elements(e.detections) WITH ORDINALITY AS item(det, ord)
    WHERE lower(det->>'class_name') = 'person'
      AND det ? 'confidence'
)
UPDATE public.camera_events AS e
SET event = people.class_name,
    confidence = people.confidence,
    person_id = e.frame_sequence::text || ':' || people.person_ordinal::text
FROM people
WHERE e.event_id = people.event_id
  AND people.person_ordinal = 1
  AND e.event IS NULL
  AND e.confidence IS NULL
  AND e.person_id IS NULL;

COMMIT;
