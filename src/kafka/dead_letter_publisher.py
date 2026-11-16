from __future__ import annotations

import base64
import json
import time

from confluent_kafka import Producer


class DeadLetterPublisher:
    """Publishes failed records while retaining their original Kafka payload."""

    def __init__(
        self,
        *,
        bootstrap_servers: str,
        topic: str,
        producer_config: dict | None = None,
        flush_timeout_seconds: float = 15.0,
        producer_factory=None,
    ):
        if not bootstrap_servers.strip():
            raise ValueError("bootstrap_servers is required")
        if not topic.strip():
            raise ValueError("Dead-letter topic is required")
        if flush_timeout_seconds <= 0:
            raise ValueError("flush_timeout_seconds must be positive")

        config = dict(producer_config or {})
        config.update({
            "bootstrap.servers": bootstrap_servers,
            "client.id": "deepstream-dead-letter-publisher",
            "acks": "all",
            "enable.idempotence": True,
        })

        factory = producer_factory or Producer
        self._producer = factory(config)
        self.topic = topic
        self.flush_timeout_seconds = flush_timeout_seconds
        self._closed = False

    @staticmethod
    def _encode(value):
        if value is None:
            return None
        return base64.b64encode(value).decode("ascii")

    def _build_envelope(self, record):
        headers = [
            {
                "name": name,
                "value_base64": self._encode(value),
            }
            for name, value in record.headers
        ]

        event_id = (
            f"{record.topic}:{record.partition}:{record.offset}"
        )

        return {
            "schema_version": 1,
            "event_id": event_id,
            "source": {
                "topic": record.topic,
                "partition": record.partition,
                "offset": record.offset,
                "timestamp_ms": record.timestamp_ms,
            },
            "model_id": (
                record.result.model_id if record.result is not None else ""
            ),
            "error": {
                "type": record.error_type or "UnknownError",
                "message": (record.error_message or "")[:2000],
            },
            "original_message": {
                "key_base64": self._encode(record.key),
                "payload_base64": self._encode(record.payload),
                "headers": headers,
            },
            "failed_at_ms": time.time_ns() // 1_000_000,
        }

    def publish_batch(self, records):
        if self._closed:
            raise RuntimeError("Dead-letter publisher is closed")

        records = list(records)
        if not records:
            return

        delivery_errors = []
        deadline = time.monotonic() + self.flush_timeout_seconds

        def delivery_callback(error, message):
            if error is not None:
                delivery_errors.append(str(error))

        for record in records:
            payload = json.dumps(
                self._build_envelope(record),
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")

            key = (
                f"{record.topic}:{record.partition}:{record.offset}"
            ).encode("utf-8")

            while True:
                try:
                    self._producer.produce(
                        self.topic,
                        key=key,
                        value=payload,
                        headers=[("content-type", b"application/json")],
                        on_delivery=delivery_callback,
                    )
                    break
                except BufferError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Dead-letter producer queue remained full"
                        )
                    self._producer.poll(0.05)

        remaining_time = max(0.0, deadline - time.monotonic())
        remaining = self._producer.flush(remaining_time)

        if remaining:
            raise TimeoutError(
                f"{remaining} dead-letter messages remain undelivered"
            )
        if delivery_errors:
            raise RuntimeError(
                "Dead-letter delivery failed: " + "; ".join(delivery_errors)
            )

    def close(self):
        if self._closed:
            return
        remaining = self._producer.flush(self.flush_timeout_seconds)
        if remaining:
            raise TimeoutError(
                f"{remaining} dead-letter messages pending at shutdown"
            )
        self._closed = True
