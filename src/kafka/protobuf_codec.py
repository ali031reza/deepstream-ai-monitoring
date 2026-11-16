from google.protobuf.message import DecodeError

from src.proto.inference_result_pb2 import InferenceResult


class ProtobufDeserializationError(ValueError):
    """Raised when a Kafka value is not a valid InferenceResult payload."""


def deserialize_inference_result(payload: bytes) -> InferenceResult:
    if not isinstance(payload, bytes):
        raise ProtobufDeserializationError("Kafka payload must be bytes")

    if not payload:
        raise ProtobufDeserializationError("Kafka payload is empty")

    result = InferenceResult()
    try:
        result.ParseFromString(payload)
    except DecodeError as exc:
        raise ProtobufDeserializationError(
            "Invalid InferenceResult Protobuf payload"
        ) from exc

    return result
