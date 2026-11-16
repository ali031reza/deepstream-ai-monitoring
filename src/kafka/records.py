from __future__ import annotations

from dataclasses import dataclass, replace

from src.proto.inference_result_pb2 import InferenceResult


@dataclass(frozen=True, slots=True)
class KafkaRecord:
    topic: str
    partition: int
    offset: int
    payload: bytes
    key: bytes | None
    headers: tuple[tuple[str, bytes | None], ...]
    timestamp_ms: int | None
    result: InferenceResult | None = None
    error_type: str | None = None
    error_message: str | None = None
    assignment_epoch: int | None = None

    @classmethod
    def from_message(cls, message) -> "KafkaRecord":
        error = message.error()
        if error is not None:
            raise ValueError(f"Cannot create record from Kafka error: {error}")

        _timestamp_type, timestamp = message.timestamp()
        raw_headers = message.headers() or []

        return cls(
            topic=message.topic(),
            partition=message.partition(),
            offset=message.offset(),
            payload=bytes(message.value() or b""),
            key=bytes(message.key()) if message.key() is not None else None,
            headers=tuple(
                (name, bytes(value) if value is not None else None)
                for name, value in raw_headers
            ),
            timestamp_ms=(
                timestamp
                if timestamp is not None and timestamp >= 0
                else None
            ),
        )

    def with_assignment_epoch(self, epoch: int | None) -> "KafkaRecord":
        return replace(self, assignment_epoch=epoch)

    def with_result(self, result: InferenceResult) -> "KafkaRecord":
        return replace(
            self,
            result=result,
            error_type=None,
            error_message=None,
        )

    def with_error(self, error: Exception) -> "KafkaRecord":
        # Preserve a decoded Protobuf result when validation (rather than
        # decoding) fails. The DLT envelope can then include its model ID.
        return replace(
            self,
            error_type=type(error).__name__,
            error_message=str(error),
        )
