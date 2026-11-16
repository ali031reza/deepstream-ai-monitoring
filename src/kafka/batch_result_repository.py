from __future__ import annotations

import hashlib
import re

from pathlib import PurePosixPath

from psycopg.types.json import Jsonb

from src.proto.inference_result_pb2 import InferenceResult


SHA256_PATTERN = re.compile(
    r"^[a-fA-F0-9]{64}$"
)


def _values_placeholder(
    row_count: int,
    column_count: int,
) -> str:

    row = (
        "("
        + ", ".join(["%s"] * column_count)
        + ")"
    )

    return ", ".join(
        [row] * row_count
    )


def _flatten(rows):

    return [
        value
        for row in rows
        for value in row
    ]


class DatabaseBatchRepository:

    def __init__(self, database):

        self.database = database

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def persist_batch(
        self,
        results: list[InferenceResult],
    ) -> None:

        if not results:
            return

        unique_results = self._deduplicate_results(
            results
        )

        with self.database.transaction() as connection:

            with connection.cursor() as cursor:

                image_ids = self._save_event_images(
                    cursor,
                    unique_results,
                )

                self._save_inference_results(
                    cursor,
                    unique_results,
                    image_ids,
                )

    # -------------------------------------------------
    # Deduplication
    # -------------------------------------------------

    @staticmethod
    def _payload_hash(result):

        payload = result.SerializeToString(
            deterministic=True
        )

        return hashlib.sha256(
            payload
        ).hexdigest()

    def _deduplicate_results(self, results):

        unique = {}

        for result in results:

            key = (
                result.model_id,
                result.frame_id,
            )

            digest = self._payload_hash(result)

            if key in unique:

                previous, previous_digest = unique[key]

                if previous_digest != digest:

                    raise ValueError(
                        "Conflicting results for the same "
                        "model/frame in one batch"
                    )

                continue

            unique[key] = (
                result,
                digest,
            )

        return [
            item[0]
            for item in unique.values()
        ]

    # -------------------------------------------------
    # Event images
    # -------------------------------------------------

    @staticmethod
    def _image_metadata(result):

        if not result.is_event:
            raise ValueError(
                "Cannot persist an image for a non-event result"
            )

        if not result.camera_id or len(result.camera_id) > 50:
            raise ValueError("Invalid camera_id")

        if not result.frame_id or len(result.frame_id) > 100:
            raise ValueError("Invalid frame_id")

        key = result.storage_key

        if (
            not key
            or key.startswith("/")
            or "\\" in key
            or any(
                part in ("", ".", "..")
                for part in key.split("/")
            )
        ):
            raise ValueError("Unsafe event image storage key")

        path = PurePosixPath(key)

        if path.is_absolute():
            raise ValueError("Absolute image paths are not allowed")

        if result.image_format.lower() not in ("jpeg", "jpg"):
            raise ValueError("Unsupported image format")

        if result.image_size_bytes <= 0:
            raise ValueError("Invalid image size")

        if not SHA256_PATTERN.fullmatch(result.image_sha256):
            raise ValueError("Invalid image SHA-256")

        if result.captured_at_ms <= 0:
            raise ValueError("Invalid capture timestamp")

        return (
            result.storage_key,
            result.image_format.lower(),
            result.image_size_bytes,
            result.image_sha256.lower(),
            result.captured_at_ms,
        )

    def _save_event_images(
        self,
        cursor,
        results,
    ):

        images = {}

        for result in results:

            if not result.is_event:
                continue

            key = (
                result.camera_id,
                result.frame_id,
            )

            metadata = self._image_metadata(
                result
            )

            if key in images and images[key] != metadata:

                raise ValueError(
                    "Conflicting event-image metadata "
                    "for the same camera/frame"
                )

            images[key] = metadata

        if not images:
            return {}

        keys = list(images.keys())

        rows = [
            (
                camera_id,
                frame_id,
                *images[(camera_id, frame_id)],
            )
            for camera_id, frame_id in keys
        ]

        placeholders = _values_placeholder(
            len(rows),
            7,
        )

        query = f"""
            INSERT INTO event_images (
                camera_id,
                frame_id,
                storage_key,
                image_format,
                file_size_bytes,
                sha256,
                captured_at_ms
            )
            VALUES {placeholders}

            ON CONFLICT (camera_id, frame_id)
            DO NOTHING
        """

        cursor.execute(
            query,
            _flatten(rows),
        )

        pair_placeholders = _values_placeholder(
            len(keys),
            2,
        )

        select_query = f"""
            SELECT
                camera_id,
                frame_id,
                id,
                storage_key,
                image_format,
                file_size_bytes,
                sha256,
                captured_at_ms

            FROM event_images

            WHERE (camera_id, frame_id)
            IN ({pair_placeholders})
        """

        cursor.execute(
            select_query,
            _flatten(keys),
        )

        existing = cursor.fetchall()

        image_ids = {}

        for row in existing:

            camera_id, frame_id = row[0], row[1]

            key = (camera_id, frame_id)

            existing_metadata = (
                row[3],
                row[4].lower(),
                row[5],
                row[6].strip().lower(),
                row[7],
            )

            if key not in images:
                continue

            if existing_metadata != images[key]:

                raise ValueError(
                    "Existing event image conflicts with "
                    "incoming image metadata"
                )

            image_ids[key] = row[2]

        if len(image_ids) != len(keys):

            raise RuntimeError(
                "Failed to retrieve all event image IDs"
            )

        return image_ids

    # -------------------------------------------------
    # Detection and timing serialization
    # -------------------------------------------------

    @staticmethod
    def _serialize_detections(result):

        return [
            {
                "class_id": detection.class_id,
                "class_name": detection.class_name,
                "confidence": detection.confidence,
                "bbox_normalized": list(
                    detection.bbox_normalized
                ),
                "synthetic": detection.synthetic,
            }
            for detection in result.detections
        ]

    @staticmethod
    def _serialize_timings(result):

        return [
            {
                "stage": timing.stage,
                "duration_ms": timing.duration_ms,
            }
            for timing in result.timings
        ]

    # -------------------------------------------------
    # Inference results
    # -------------------------------------------------

    def _save_inference_results(
        self,
        cursor,
        results,
        image_ids,
    ):

        rows = []

        for result in results:

            if not result.camera_id or len(result.camera_id) > 50:
                raise ValueError("Invalid camera_id")

            if not result.frame_id or len(result.frame_id) > 100:
                raise ValueError("Invalid frame_id")

            if not result.model_id or len(result.model_id) > 50:
                raise ValueError("Invalid model_id")

            if not result.status or len(result.status) > 50:
                raise ValueError("Invalid status")

            if not result.result_id:
                raise ValueError("Missing result_id")

            if not result.source_session_id:
                raise ValueError("Missing source_session_id")

            if not result.worker_id:
                raise ValueError("Missing worker_id")

            event_image_id = None

            if result.is_event:

                event_image_id = image_ids[
                    (
                        result.camera_id,
                        result.frame_id,
                    )
                ]

            digest = self._payload_hash(result)

            rows.append((
                result.camera_id,
                result.frame_id,
                result.model_id,
                result.processed_at_ms,
                result.status,
                result.is_event,
                result.event_type if result.is_event else None,
                event_image_id,
                result.source_session_id,
                result.result_id,
                result.worker_id,
                result.frame_sequence,
                result.gpu_id,
                result.frame_width,
                result.frame_height,
                result.model_input_size,
                result.captured_at_ms,
                result.schema_version,
                digest,
                Jsonb(self._serialize_detections(result)),
                Jsonb(self._serialize_timings(result)),
            ))

        if not rows:
            return

        placeholders = _values_placeholder(
            len(rows),
            21,
        )

        insert_query = f"""
            INSERT INTO inference_results (
                camera_id,
                frame_id,
                model_id,
                processed_at_ms,
                status,
                is_event,
                event_type,
                event_image_id,
                source_session_id,
                result_id,
                worker_id,
                frame_sequence,
                gpu_id,
                frame_width,
                frame_height,
                model_input_size,
                captured_at_ms,
                schema_version,
                payload_sha256,
                detections,
                timings
            )

            VALUES {placeholders}

            ON CONFLICT (model_id, frame_id)
            DO NOTHING
        """

        cursor.execute(
            insert_query,
            _flatten(rows),
        )

        keys = [
            (
                result.model_id,
                result.frame_id,
            )
            for result in results
        ]

        select_placeholders = _values_placeholder(
            len(keys),
            2,
        )

        select_query = f"""
            SELECT
                id,
                model_id,
                frame_id,
                result_id,
                payload_sha256

            FROM inference_results

            WHERE (model_id, frame_id)
            IN ({select_placeholders})
        """

        cursor.execute(
            select_query,
            _flatten(keys),
        )

        existing = cursor.fetchall()

        existing_by_key = {
            (row[1], row[2]): row
            for row in existing
        }

        for result in results:

            key = (
                result.model_id,
                result.frame_id,
            )

            row = existing_by_key.get(key)

            if row is None:

                raise RuntimeError(
                    f"Could not retrieve inference result {key}"
                )

            expected_hash = self._payload_hash(
                result
            )

            existing_result_id = row[3]

            existing_hash = (
                row[4].strip()
                if row[4]
                else ""
            )

            if (
                existing_result_id != result.result_id
                or existing_hash != expected_hash
            ):

                raise ValueError(
                    "Conflicting existing inference result "
                    f"for model/frame {key}"
                )