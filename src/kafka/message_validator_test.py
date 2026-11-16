import unittest

from src.kafka.message_validator import (
    MessageValidationError,
    validate_inference_result,
)
from src.proto.inference_result_pb2 import InferenceResult


def build_valid_result():
    result = InferenceResult(
        camera_id="camera-01",
        frame_id="camera-01-frame-001",
        model_id="mock_model_a",
        processed_at_ms=1794744600000,
        captured_at_ms=1794744599000,
        status="success",
        source_session_id="session-001",
        result_id="result-camera-01-model-a-001",
        worker_id="mock_model_a-worker-01",
        frame_sequence=1,
        gpu_id=0,
        frame_width=1920,
        frame_height=1080,
        model_input_size=640,
        schema_version="1.0",
    )
    result.detections.add(
        class_id=0,
        class_name="person",
        confidence=0.95,
        bbox_normalized=[0.1, 0.2, 0.6, 0.9],
        synthetic=True,
    )
    result.timings.add(stage="inference", duration_ms=4.5)
    return result


class MessageValidatorTest(unittest.TestCase):
    def test_valid_result_is_accepted(self):
        validate_inference_result(
            build_valid_result(),
            allowed_model_ids={"mock_model_a", "mock_model_b"},
        )

    def test_missing_camera_id_is_rejected(self):
        result = build_valid_result()
        result.camera_id = ""
        with self.assertRaises(MessageValidationError):
            validate_inference_result(result)

    def test_missing_result_id_is_rejected(self):
        result = build_valid_result()
        result.result_id = ""
        with self.assertRaises(MessageValidationError):
            validate_inference_result(result)

    def test_missing_schema_version_is_rejected(self):
        result = build_valid_result()
        result.schema_version = ""
        with self.assertRaises(MessageValidationError):
            validate_inference_result(result)

    def test_unsupported_model_is_rejected(self):
        result = build_valid_result()
        result.model_id = "unknown_model"
        with self.assertRaises(MessageValidationError):
            validate_inference_result(
                result,
                allowed_model_ids={"mock_model_a", "mock_model_b"},
            )

    def test_invalid_confidence_is_rejected(self):
        result = build_valid_result()
        result.detections[0].confidence = 1.5
        with self.assertRaises(MessageValidationError):
            validate_inference_result(result)

    def test_invalid_bounding_box_is_rejected(self):
        result = build_valid_result()
        result.detections[0].bbox_normalized[:] = [0.8, 0.2, 0.3, 0.9]
        with self.assertRaises(MessageValidationError):
            validate_inference_result(result)

    def test_event_without_image_metadata_is_rejected(self):
        result = build_valid_result()
        result.is_event = True
        result.event_type = "person_detected"
        with self.assertRaises(MessageValidationError):
            validate_inference_result(result)

    def test_event_with_complete_image_metadata_is_accepted(self):
        result = build_valid_result()
        result.is_event = True
        result.event_type = "person_detected"
        result.storage_key = "camera-01/frame-001.jpg"
        result.image_format = "jpeg"
        result.image_size_bytes = 1024
        result.image_sha256 = "a" * 64
        validate_inference_result(result)

    def test_invalid_timing_is_rejected(self):
        result = build_valid_result()
        result.timings[0].duration_ms = -1.0
        with self.assertRaises(MessageValidationError):
            validate_inference_result(result)

    def test_frame_dimension_mismatch_is_rejected(self):
        result = build_valid_result()
        result.frame_height = 0
        with self.assertRaises(MessageValidationError):
            validate_inference_result(result)

    def test_no_detections_is_valid(self):
        result = build_valid_result()
        result.ClearField("detections")
        validate_inference_result(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
