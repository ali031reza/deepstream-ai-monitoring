
"""Reliable Kafka consumer for DeepStream inference results."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable

from confluent_kafka import Consumer, KafkaError


logger = logging.getLogger(__name__)


class KafkaConsumerError(RuntimeError):
    """Raised when Kafka consumption or acknowledgement fails."""


@dataclass(frozen=True)
class ConsumerConfig:
    bootstrap_servers: str
    group_id: str
    topics: tuple[str, ...]

    auto_offset_reset: str = "earliest"
    poll_timeout_seconds: float = 1.0
    session_timeout_ms: int = 45000
    max_poll_interval_ms: int = 300000

    def __post_init__(self) -> None:
        if not self.bootstrap_servers.strip():
            raise ValueError("bootstrap_servers is required")

        if not self.group_id.strip():
            raise ValueError("group_id is required")

        if not self.topics:
            raise ValueError("At least one topic is required")

        if any(not topic.strip() for topic in self.topics):
            raise ValueError("Topic names cannot be empty")

        if len(set(self.topics)) != len(self.topics):
            raise ValueError("Duplicate topics are not allowed")

        if self.auto_offset_reset not in ("earliest", "latest"):
            raise ValueError(
                "auto_offset_reset must be earliest or latest"
            )

        if self.poll_timeout_seconds <= 0:
            raise ValueError("Poll timeout must be positive")

        if self.session_timeout_ms <= 0:
            raise ValueError("Session timeout must be positive")

        if self.max_poll_interval_ms <= 0:
            raise ValueError("Max poll interval must be positive")

    def to_client_config(self) -> dict:
        return {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            "client.id": "deepstream-results-consumer",

            # Explicit acknowledgement management.
            "enable.auto.commit": False,
            "enable.auto.offset.store": False,

            "auto.offset.reset": self.auto_offset_reset,
            "session.timeout.ms": self.session_timeout_ms,
            "max.poll.interval.ms": self.max_poll_interval_ms,
        }


class KafkaResultConsumer:
    """
    Consume inference results from multiple Kafka topics.

    A message is acknowledged only after its handler returns
    successfully.

    The handler must finish all required processing, including
    database persistence or successful dead-letter publication,
    before returning.
    """

    def __init__(
        self,
        config: ConsumerConfig,
        consumer_factory: Callable | None = None,
    ) -> None:

        self.config = config

        factory = consumer_factory or Consumer

        self._consumer = factory(config.to_client_config())

        self._subscribed = False
        self._closed = False

        self._lock = threading.Lock()

    def subscribe(self) -> None:
        """Subscribe to all configured result topics."""

        with self._lock:
            if self._closed:
                raise KafkaConsumerError(
                    "Cannot subscribe: consumer is closed"
                )

            if self._subscribed:
                return

            self._consumer.subscribe(list(self.config.topics))
            self._subscribed = True

        logger.info(
            "Subscribed to Kafka topics: %s",
            self.config.topics,
        )

    def poll(self, timeout: float | None = None):
        """Retrieve one Kafka message."""

        if self._closed:
            raise KafkaConsumerError("Consumer is closed")

        if not self._subscribed:
            raise KafkaConsumerError(
                "Consumer must subscribe before polling"
            )

        if timeout is None:
            timeout = self.config.poll_timeout_seconds

        if timeout <= 0:
            raise ValueError("Poll timeout must be positive")

        message = self._consumer.poll(timeout)

        if message is None:
            return None

        error = message.error()

        if error is not None:

            if error.code() == KafkaError._PARTITION_EOF:
                return None

            raise KafkaConsumerError(
                f"Kafka polling failed: {error}"
            )

        return message

    def commit(self, message) -> None:
        """
        Commit the offset of one successfully processed message.

        The synchronous commit ensures the acknowledgement result
        is checked before processing continues.
        """

        if self._closed:
            raise KafkaConsumerError(
                "Cannot commit: consumer is closed"
            )

        try:
            result = self._consumer.commit(
                message=message,
                asynchronous=False,
            )

        except Exception as exc:
            raise KafkaConsumerError(
                f"Failed to commit Kafka offset: {exc}"
            ) from exc

        # Check per-partition commit errors when returned.
        if result is not None:
            for partition in result:
                error = getattr(partition, "error", None)

                if callable(error):
                    error = error()

                if error is not None:
                    raise KafkaConsumerError(
                        f"Offset commit failed: {error}"
                    )

        logger.debug(
            "Committed topic=%s partition=%s offset=%s",
            message.topic(),
            message.partition(),
            message.offset(),
        )

    def process_next(
        self,
        handler: Callable,
        timeout: float | None = None,
    ) -> bool:
        """
        Process one message and commit only after success.

        Returns:
            True: A message was successfully processed and committed.
            False: No message was available.
        """

        message = self.poll(timeout)

        if message is None:
            return False

        # If this raises, the offset is NOT committed.
        # The caller should stop this consumer instance.
        handler(message)

        # Commit only after successful handler completion.
        self.commit(message)

        return True

    def run(
        self,
        handler: Callable,
        stop_event: threading.Event,
    ) -> None:
        """
        Run until shutdown is requested.

        Processing failures propagate to the caller. The consumer
        closes without acknowledging the failed message.
        """

        self.subscribe()

        try:
            while not stop_event.is_set():
                self.process_next(handler)

        finally:
            self.close()

    def close(self) -> None:
        """Close the Kafka consumer safely."""

        with self._lock:
            if self._closed:
                return

            self._closed = True

        try:
            self._consumer.close()
            logger.info("Kafka consumer closed")

        except Exception:
            logger.exception("Error while closing Kafka consumer")

    def __enter__(self):
        self.subscribe()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
