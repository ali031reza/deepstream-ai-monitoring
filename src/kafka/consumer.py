from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Iterable

from confluent_kafka import Consumer, KafkaError, TopicPartition


logger = logging.getLogger(__name__)


class KafkaConsumerError(RuntimeError):
    """Raised when Kafka polling, assignment, or acknowledgement is unsafe."""


@dataclass(frozen=True)
class ConsumerConfig:
    bootstrap_servers: str
    group_id: str
    topics: tuple[str, ...]
    auto_offset_reset: str = "earliest"
    poll_timeout_seconds: float = 0.5
    session_timeout_ms: int = 45000
    max_poll_interval_ms: int = 900000

    def __post_init__(self):
        if not self.bootstrap_servers.strip():
            raise ValueError("bootstrap_servers is required")
        if not self.group_id.strip():
            raise ValueError("group_id is required")
        if not self.topics or any(not t.strip() for t in self.topics):
            raise ValueError("At least one valid topic is required")
        if len(set(self.topics)) != len(self.topics):
            raise ValueError("Duplicate topics are not allowed")
        if self.auto_offset_reset not in ("earliest", "latest"):
            raise ValueError("auto_offset_reset must be earliest or latest")
        if self.poll_timeout_seconds <= 0:
            raise ValueError("poll_timeout_seconds must be positive")
        if self.session_timeout_ms <= 0 or self.max_poll_interval_ms <= 0:
            raise ValueError("Kafka poll/session intervals must be positive")

    def to_client_config(self) -> dict:
        return {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": self.group_id,
            "client.id": "deepstream-results-consumer",
            "enable.auto.commit": False,
            "enable.auto.offset.store": False,
            "auto.offset.reset": self.auto_offset_reset,
            "session.timeout.ms": self.session_timeout_ms,
            "max.poll.interval.ms": self.max_poll_interval_ms,
            "isolation.level": "read_committed",
            "partition.assignment.strategy": "cooperative-sticky",
        }


class KafkaResultConsumer:
    """
    Manual-commit consumer. Poll, pause, resume and commit MUST be called on
    its owning thread. The assignment epoch fences completions from partitions
    revoked/lost during asynchronous persistence. Replayed DB/DLT operations
    must be idempotent or at-least-once safe.
    """

    def __init__(
        self,
        config: ConsumerConfig,
        consumer_factory: Callable | None = None,
    ):
        self.config = config
        factory = consumer_factory or Consumer
        self._consumer = factory(config.to_client_config())
        self._subscribed = False
        self._closed = False
        self._paused_partitions = {}
        self._epoch = 0
        self.assignment_ready = threading.Event()
        self.rebalances = 0
        self.lost_assignments = 0

    @property
    def rebalance_epoch(self):
        return self._epoch

    @staticmethod
    def _key(partition):
        return partition.topic, partition.partition

    def _on_assign(self, consumer, partitions):
        logger.info("Kafka partitions assigned: %s", partitions)
        # librdkafka performs the assignment when the callback does not
        # explicitly call assign()/incremental_assign().

    def _on_revoke(self, consumer, partitions):
        self._epoch += 1
        self.rebalances += 1
        for partition in partitions:
            self._paused_partitions.pop(self._key(partition), None)
        logger.warning(
            "Kafka partitions revoked; epoch=%d, awaiting replay of any "
            "in-flight records: %s", self._epoch, partitions,
        )

    def _on_lost(self, consumer, partitions):
        self._epoch += 1
        self.lost_assignments += 1
        for partition in partitions:
            self._paused_partitions.pop(self._key(partition), None)
        logger.error(
            "Kafka assignment lost; epoch=%d, stale commits forbidden: %s",
            self._epoch, partitions,
        )

    def subscribe(self):
        if self._closed:
            raise KafkaConsumerError("Cannot subscribe: consumer is closed")
        if self._subscribed:
            return
        self._consumer.subscribe(
            list(self.config.topics),
            on_assign=self._on_assign,
            on_revoke=self._on_revoke,
            on_lost=self._on_lost,
        )
        self._subscribed = True
        logger.info("Subscribed to Kafka topics: %s", self.config.topics)

    def poll(self, timeout: float | None = None):
        if self._closed:
            raise KafkaConsumerError("Consumer is closed")
        if not self._subscribed:
            raise KafkaConsumerError("Consumer must subscribe before polling")
        if timeout is None:
            timeout = self.config.poll_timeout_seconds
        if timeout < 0:
            raise ValueError("Poll timeout cannot be negative")

        message = self._consumer.poll(timeout)
        # Signal only after poll returns from assignment callbacks, when
        # librdkafka has had a chance to finish applying the assignment.
        if self._consumer.assignment():
            self.assignment_ready.set()
        if message is None:
            return None
        error = message.error()
        if error is not None:
            if error.code() == KafkaError._PARTITION_EOF:
                return None
            raise KafkaConsumerError(f"Kafka polling failed: {error}")
        return message

    def service_events(self, timeout=0.05):
        """
        Keep consumer group callbacks/heartbeats active while paused or
        draining. A data message returned despite pause is NOT ignored:
        fail closed so it remains uncommitted and is replayed.
        """
        unexpected = self.poll(timeout)
        if unexpected is not None:
            raise KafkaConsumerError(
                "Kafka delivered a data record while service was paused "
                f"(topic={unexpected.topic()} partition={unexpected.partition()} "
                f"offset={unexpected.offset()}); stopping without acknowledging it"
            )

    def pause(self):
        if self._closed:
            return
        current = {self._key(p): p for p in self._consumer.assignment()}
        new = [p for key, p in current.items() if key not in self._paused_partitions]
        if new:
            self._consumer.pause(new)
            self._paused_partitions.update({self._key(p): p for p in new})
            logger.warning("Paused Kafka partitions: %s", new)
        # Removed/revoked partitions should never remain marked as paused.
        for key in list(self._paused_partitions):
            if key not in current:
                del self._paused_partitions[key]

    def resume(self):
        if self._closed or not self._paused_partitions:
            return
        current = {self._key(p): p for p in self._consumer.assignment()}
        to_resume = [current[key] for key in self._paused_partitions if key in current]
        if to_resume:
            self._consumer.resume(to_resume)
        self._paused_partitions.clear()
        logger.info("Resumed Kafka partitions")

    def commit(self, message):
        if self._closed:
            raise KafkaConsumerError("Cannot commit using a closed consumer")
        result = self._consumer.commit(message=message, asynchronous=False)
        self._check_commit_result(result)

    def commit_batch(self, records: Iterable):
        if self._closed:
            raise KafkaConsumerError("Cannot commit using a closed consumer")

        next_offsets = {}
        assigned = {self._key(p) for p in self._consumer.assignment()}
        for record in records:
            key = (record.topic, record.partition)
            record_epoch = getattr(record, "assignment_epoch", None)
            if record_epoch is not None and record_epoch != self._epoch:
                raise KafkaConsumerError(
                    "Refusing stale Kafka commit after partition rebalance: "
                    f"record_epoch={record_epoch} current_epoch={self._epoch}"
                )
            if key not in assigned:
                raise KafkaConsumerError(
                    f"Refusing Kafka commit for unassigned partition {key}"
                )
            next_offsets[key] = max(
                next_offsets.get(key, 0), record.offset + 1
            )

        if not next_offsets:
            return
        offsets = [
            TopicPartition(topic, partition, offset)
            for (topic, partition), offset in sorted(next_offsets.items())
        ]
        result = self._consumer.commit(offsets=offsets, asynchronous=False)
        self._check_commit_result(result)
        logger.info("Committed offsets for %d partition(s)", len(offsets))

    @staticmethod
    def _check_commit_result(result):
        if result is None:
            return
        for partition in result:
            error = getattr(partition, "error", None)
            if callable(error):
                error = error()
            if error is not None:
                raise KafkaConsumerError(f"Kafka offset commit failed: {error}")

    def process_next(self, handler, timeout=None) -> bool:
        message = self.poll(timeout)
        if message is None:
            return False
        handler(message)
        self.commit(message)
        return True

    def run(self, handler, stop_event):
        self.subscribe()
        try:
            while not stop_event.is_set():
                self.process_next(handler)
        finally:
            self.close()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._consumer.close()
            logger.info("Kafka consumer closed")
        except Exception:
            logger.exception("Error while closing Kafka consumer")
