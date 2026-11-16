"""Bounded-queue, asynchronous Kafka publisher for one model only."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from confluent_kafka import Producer

from .model_result_message import ModelResultMessage


@dataclass(slots=True)
class _QueuedMessage:
    payload: bytes
    key: bytes
    frame_id: int


class AsyncKafkaPublisher:
    """One instance owns one model's queue, Kafka producer and worker thread."""

    def __init__(
        self,
        *,
        model_id: str,
        topic: str,
        producer_config: dict[str, Any],
        queue_maxsize: int = 128,
        max_batch_messages: int = 32,
        batch_flush_interval_ms: int = 100,
        enqueue_timeout_seconds: float = 5.0,
        produce_retry_seconds: float = 120.0,
        shutdown_timeout_seconds: float = 30.0,
    ) -> None:
        if not model_id:
            raise ValueError("model_id must not be empty")
        if not topic:
            raise ValueError("topic must not be empty")
        if queue_maxsize <= 0 or max_batch_messages <= 0:
            raise ValueError("queue_maxsize and max_batch_messages must be positive")
        if batch_flush_interval_ms <= 0:
            raise ValueError("batch_flush_interval_ms must be positive")

        self.model_id = model_id
        self.topic = topic
        self.queue_maxsize = queue_maxsize
        self.max_batch_messages = max_batch_messages
        self.batch_flush_interval_seconds = batch_flush_interval_ms / 1000.0
        self.enqueue_timeout_seconds = enqueue_timeout_seconds
        self.produce_retry_seconds = produce_retry_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds

        config = dict(producer_config)
        if "bootstrap.servers" not in config:
            raise ValueError("producer_config must include bootstrap.servers")
        config.setdefault("client.id", f"deepstream-{model_id}-publisher")

        self._producer = Producer(config)
        self._queue: queue.Queue[_QueuedMessage] = queue.Queue(
            maxsize=queue_maxsize
        )
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = False
        self._accepting = False
        self._closed = False
        self._fatal_error: str | None = None

        self._stats: dict[str, Any] = {
            "model_id": model_id,
            "topic": topic,
            "enqueued": 0,
            "produce_accepted": 0,
            "delivered": 0,
            "delivery_failures": 0,
            "publish_failures": 0,
            "enqueue_timeouts": 0,
            "batches_submitted": 0,
            "max_queue_depth": 0,
            "worker_errors": 0,
            "shutdown_pending": 0,
            "last_errors": [],
        }

    def start(self) -> "AsyncKafkaPublisher":
        with self._state_lock:
            if self._closed:
                raise RuntimeError("publisher is already closed")
            if self._started:
                return self
            self._accepting = True
            self._started = True
            self._thread = threading.Thread(
                target=self._worker_loop,
                name=f"kafka-publisher-{self.model_id}",
                daemon=False,
            )
            self._thread.start()
        return self

    def publish(
        self,
        message: ModelResultMessage,
        *,
        timeout: float | None = None,
    ) -> None:
        if not isinstance(message, ModelResultMessage):
            raise TypeError("publish expects a ModelResultMessage")
        if message.model_id != self.model_id:
            raise ValueError(
                f"{self.model_id} publisher cannot publish "
                f"a {message.model_id} message"
            )

        item = _QueuedMessage(
            payload=message.to_json_bytes(),
            key=f"{message.source_id}:{message.frame_id}".encode("utf-8"),
            frame_id=message.frame_id,
        )
        wait_seconds = (
            self.enqueue_timeout_seconds if timeout is None else timeout
        )

        with self._state_lock:
            if not self._started or not self._accepting:
                raise RuntimeError("publisher is not accepting messages")
            try:
                self._queue.put(item, timeout=wait_seconds)
            except queue.Full as exc:
                self._increment("enqueue_timeouts")
                raise TimeoutError(
                    f"{self.model_id} result queue remained full "
                    f"for {wait_seconds:.3f} seconds"
                ) from exc

        with self._stats_lock:
            self._stats["enqueued"] += 1
            self._stats["max_queue_depth"] = max(
                self._stats["max_queue_depth"], self._queue.qsize()
            )

    def _increment(self, field: str, amount: int = 1) -> None:
        with self._stats_lock:
            self._stats[field] += amount

    def _record_error(self, message: str) -> None:
        with self._stats_lock:
            errors = self._stats["last_errors"]
            errors.append(message)
            del errors[:-20]

    def _delivery_callback(self, error: Any, kafka_message: Any) -> None:
        if error is not None:
            self._increment("delivery_failures")
            self._record_error(f"delivery failed: {error}")
        else:
            self._increment("delivered")

    def _produce_with_backpressure(self, item: _QueuedMessage) -> None:
        deadline = time.monotonic() + self.produce_retry_seconds
        while True:
            try:
                self._producer.produce(
                    self.topic,
                    key=item.key,
                    value=item.payload,
                    on_delivery=self._delivery_callback,
                )
                self._increment("produce_accepted")
                return
            except BufferError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Kafka producer local queue remained full"
                    )
                self._producer.poll(0.05)

    def _submit_batch(self, batch: list[_QueuedMessage]) -> None:
        accepted_in_batch = 0
        for item in batch:
            try:
                self._produce_with_backpressure(item)
                accepted_in_batch += 1
            except Exception as exc:
                self._increment("publish_failures")
                self._record_error(
                    f"frame_id={item.frame_id}: {type(exc).__name__}: {exc}"
                )
            finally:
                self._queue.task_done()

        if accepted_in_batch:
            self._increment("batches_submitted")
        self._producer.poll(0)

    def _worker_loop(self) -> None:
        try:
            while not self._stop_event.is_set() or not self._queue.empty():
                try:
                    first = self._queue.get(
                        timeout=min(self.batch_flush_interval_seconds, 0.1)
                    )
                except queue.Empty:
                    self._producer.poll(0)
                    continue

                batch = [first]
                batch_deadline = (
                    time.monotonic() + self.batch_flush_interval_seconds
                )

                while len(batch) < self.max_batch_messages:
                    remaining = batch_deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        batch.append(self._queue.get(timeout=remaining))
                    except queue.Empty:
                        break

                self._submit_batch(batch)

        except Exception as exc:
            self._increment("worker_errors")
            self._record_error(
                f"worker failed: {type(exc).__name__}: {exc}"
            )
            self._fatal_error = f"{type(exc).__name__}: {exc}"

            # Account for queued work if the worker cannot continue.
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._increment("publish_failures")
                    self._queue.task_done()

        finally:
            try:
                pending = self._producer.flush(self.shutdown_timeout_seconds)
                if pending:
                    self._increment("shutdown_pending", pending)
                    self._record_error(
                        f"{pending} Kafka message(s) remained undelivered "
                        "when shutdown timeout expired"
                    )
            except Exception as exc:
                self._increment("worker_errors")
                self._record_error(
                    f"flush failed: {type(exc).__name__}: {exc}"
                )
                self._fatal_error = self._fatal_error or str(exc)

    def close(self) -> dict[str, Any]:
        with self._state_lock:
            if self._closed:
                return self.stats()
            self._accepting = False
            self._stop_event.set()
            thread = self._thread
            if not self._started:
                self._closed = True
                return self.stats()

        assert thread is not None
        thread.join(timeout=self.shutdown_timeout_seconds + 5.0)
        if thread.is_alive():
            raise TimeoutError(
                f"{self.model_id} publisher worker did not stop in time"
            )

        with self._state_lock:
            self._closed = True

        result = self.stats()
        if self._fatal_error:
            raise RuntimeError(
                f"{self.model_id} publisher worker failed: {self._fatal_error}"
            )
        return result

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            snapshot = dict(self._stats)
            snapshot["last_errors"] = list(self._stats["last_errors"])
        snapshot["queue_depth"] = self._queue.qsize()
        snapshot["outstanding_after_produce"] = max(
            0,
            snapshot["produce_accepted"]
            - snapshot["delivered"]
            - snapshot["delivery_failures"],
        )
        return snapshot
