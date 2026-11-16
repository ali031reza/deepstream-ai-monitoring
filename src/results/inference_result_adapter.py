
from datetime import datetime
import math

from src.proto.inference_result_pb2 import InferenceResult
from src.results.model_result_message import ModelResultMessage


class InferenceResultAdapter:
    """Convert DeepStream model results into backend Protobuf messages."""

    @staticmethod
    def _required_text(value, name):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _processed_timestamp(value):
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)

        if parsed.tzinfo is None:
            raise ValueError("processed_at_utc must include a timezone")

        timestamp = int(parsed.timestamp() * 1000)

        if timestamp <= 0:
            raise ValueError("Invalid processing timestamp")

        return timestamp

    @classmethod
    def to_protobuf(
        cls,
        message: ModelResultMessage,
        *,
        source_session_id: str,
        worker_id: str,
        captured_at_ms: int,
        is_event: bool = False,
        event_type: str = "",
        event_image=None,
    ) -> InferenceResult:

        if not isinstance(message, ModelResultMessage):
            raise TypeError("Expected ModelResultMessage")

        source_session_id = cls._required_text(
            source_session_id,
            "source_session_id",
        )

        worker_id = cls._required_text(
            worker_id,
            "worker_id",
        )

        if isinstance(captured_at_ms, bool) or not isinstance(
            captured_at_ms, int
        ) or captured_at_ms <= 0:
            raise ValueError("Invalid captured_at_ms")

        if is_event:
            event_type = cls._required_text(event_type, "event_type")

            if event_image is None:
                raise ValueError(
                    "An event result requires a stored event image"
                )
        elif event_type or event_image is not None:
            raise ValueError(
                "Non-event results cannot contain event metadata"
            )

        frame_id = (
            f"{source_session_id}:{message.frame_id:012d}"
        )

        if len(frame_id) > 100:
            raise ValueError("Generated frame_id exceeds 100 characters")

        result = InferenceResult(
            camera_id=message.source_id,
            frame_id=frame_id,
            model_id=message.model_id,

            processed_at_ms=cls._processed_timestamp(
                message.processed_at_utc
            ),

            captured_at_ms=captured_at_ms,

            status="processed",
            is_event=is_event,
            event_type=event_type if is_event else "",

            source_session_id=source_session_id,
            result_id=message.message_id,
            worker_id=worker_id,

            frame_sequence=message.frame_id,

            gpu_id=message.gpu_id,
            frame_width=message.frame_width,
            frame_height=message.frame_height,
            model_input_size=message.model_input_size or 0,

            schema_version=message.schema_version,
        )

        # Add detection information.
        for index, raw in enumerate(message.detections):

            class_id = raw.get("class_id", -1)

            if isinstance(class_id, bool) or not isinstance(class_id, int):
                raise ValueError(
                    f"detections[{index}].class_id must be an integer"
                )

            confidence = float(raw.get("confidence", 0.0))

            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError(
                    f"Invalid confidence in detection {index}"
                )

            bbox = raw.get("bbox_normalized", [])

            if not isinstance(bbox, (list, tuple)):
                raise ValueError("bbox_normalized must be a sequence")

            if bbox and len(bbox) != 4:
                raise ValueError(
                    "bbox_normalized must contain four coordinates"
                )

            coordinates = [float(value) for value in bbox]

            if any(
                not math.isfinite(value) or not 0 <= value <= 1
                for value in coordinates
            ):
                raise ValueError(
                    f"Invalid normalized bounding box in detection {index}"
                )

            synthetic = raw.get("synthetic", False)

            if not isinstance(synthetic, bool):
                raise ValueError("synthetic must be a boolean")

            detection = result.detections.add()

            detection.class_id = class_id
            detection.class_name = str(raw.get("class_name", ""))
            detection.confidence = confidence
            detection.bbox_normalized.extend(coordinates)
            detection.synthetic = synthetic

        # Add inference timing information.
        for stage, duration in message.timings_ms.items():

            result.timings.add(
                stage=stage,
                duration_ms=float(duration),
            )

        # Attach image metadata only after successful storage.
        if is_event:

            result.storage_key = event_image.storage_key
            result.image_format = event_image.image_format
            result.image_size_bytes = event_image.file_size_bytes
            result.image_sha256 = event_image.sha256

            if event_image.captured_at_ms != captured_at_ms:
                raise ValueError(
                    "Event image timestamp does not match frame timestamp"
                )

        return result
