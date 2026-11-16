import unittest

from src.kafka.protobuf_codec import (
    ProtobufDeserializationError,
    deserialize_inference_result,
)
from src.proto.inference_result_pb2 import InferenceResult


class ProtobufCodecTest(unittest.TestCase):
    def test_valid_payload(self):
        original = InferenceResult(
            camera_id="camera-01",
            frame_id="frame-01",
            model_id="mock_model_a",
            result_id="result-01",
            status="processed",
        )
        decoded = deserialize_inference_result(
            original.SerializeToString()
        )
        self.assertEqual(decoded.camera_id, original.camera_id)
        self.assertEqual(decoded.result_id, original.result_id)

    def test_empty_payload(self):
        with self.assertRaises(ProtobufDeserializationError):
            deserialize_inference_result(b"")

    def test_invalid_payload(self):
        with self.assertRaises(ProtobufDeserializationError):
            deserialize_inference_result(b"\xff")

    def test_non_bytes_payload(self):
        with self.assertRaises(ProtobufDeserializationError):
            deserialize_inference_result("invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
