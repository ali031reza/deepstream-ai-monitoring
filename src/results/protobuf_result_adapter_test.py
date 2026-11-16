
import unittest

from src.proto.inference_result_pb2 import InferenceResult
from src.results.protobuf_result_adapter import (
    build_inference_result,
)


class ProtobufResultAdapterTest(unittest.TestCase):

    def make_output(self, model_id, frame_id=105):
        return {
            "model_id": model_id,
            "frame_id": frame_id,
            "gpu_id": 0,
            "input_frame_shape": [720, 1280, 3],
            "model_input_size": 320,
            "detections": [
                {
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.95,
                    "bbox_normalized": [
                        0.1, 0.2, 0.3, 0.4
                    ],
                    "synthetic": True,
                }
            ],
            "timings_ms": {
                "preprocessing_gpu_ms": 0.45,
                "inference_gpu_ms": 0.75,
                "postprocessing_gpu_ms": 0.25,
            },
        }

    def build(self, model_id):
        return build_inference_result(
            self.make_output(model_id),
            camera_id="camera_0",
            source_session_id="session_001",
            frame_sequence=105,
            captured_at_ms=1794744599900,
            worker_id=f"{model_id}-worker-01",
        )

    def test_model_a_and_b_share_frame_identity(self):
        a = self.build("mock_model_a")
        b = self.build("mock_model_b")

        self.assertEqual(a.camera_id, b.camera_id)
        self.assertEqual(a.source_session_id, b.source_session_id)
        self.assertEqual(a.frame_id, b.frame_id)
        self.assertEqual(a.frame_sequence, b.frame_sequence)

        self.assertNotEqual(a.model_id, b.model_id)
        self.assertNotEqual(a.result_id, b.result_id)

    def test_detection_and_timing_mapping(self):
        result = self.build("mock_model_a")

        self.assertEqual(len(result.detections), 1)
        self.assertEqual(
            result.detections[0].class_name, "person"
        )
        self.assertAlmostEqual(
            result.detections[0].confidence, 0.95
        )

        self.assertEqual(len(result.timings), 3)

    def test_protobuf_serialization_round_trip(self):
        original = self.build("mock_model_a")

        payload = original.SerializeToString()

        decoded = InferenceResult()
        decoded.ParseFromString(payload)

        self.assertEqual(decoded.result_id, original.result_id)
        self.assertEqual(decoded.frame_id, original.frame_id)
        self.assertEqual(decoded.model_id, original.model_id)
        self.assertEqual(
            decoded.detections[0].class_name, "person"
        )

    def test_result_id_is_stable_for_retries(self):
        first = self.build("mock_model_a")
        second = self.build("mock_model_a")

        self.assertEqual(first.result_id, second.result_id)

    def test_model_frame_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            build_inference_result(
                self.make_output("mock_model_a", frame_id=104),
                camera_id="camera_0",
                source_session_id="session_001",
                frame_sequence=105,
                captured_at_ms=1794744599900,
                worker_id="mock_model_a-worker-01",
            )

    def test_event_without_image_reference_is_rejected(self):
        output = self.make_output("mock_model_a")
        output["is_event"] = True
        output["event_type"] = "test_event"

        with self.assertRaises(ValueError):
            build_inference_result(
                output,
                camera_id="camera_0",
                source_session_id="session_001",
                frame_sequence=105,
                captured_at_ms=1794744599900,
                worker_id="mock_model_a-worker-01",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
