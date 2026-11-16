
"""Asynchronous Kafka publisher for binary Protobuf inference results."""

from __future__ import annotations

import math
import queue
import threading
import time

from dataclasses import dataclass
from typing import Any

from confluent_kafka import Producer

from src.proto.inference_result_pb2 import InferenceResult


@dataclass(slots=True)
class _QueuedMessage:
    payload: bytes
    key: bytes
    frame_id: str


class AsyncProtobufKafkaPublisher:
    """One independent publisher instance per AI model."""

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
    ):

        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must not be empty")

        if not isinstance(topic, str) or not topic.strip():
            raise ValueError("topic must not be empty")

        for name, value in (
            ("queue_maxsize", queue_maxsize),
            ("max_batch_messages", max_batch_messages),
            ("batch_flush_interval_ms", batch_flush_interval_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        for name, value in (
            ("enqueue_timeout_seconds", enqueue_timeout_seconds),
            ("produce_retry_seconds", produce_retry_seconds),
            ("shutdown_timeout_seconds", shutdown_timeout_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive and finite")

        self.model_id = model_id.strip()
        self.topic = topic.strip()

        self.max_batch_messages = max_batch_messages
        self.batch_flush_interval_seconds = (
            batch_flush_interval_ms / 1000.0
        )

        self.enqueue_timeout_seconds = float(enqueue_timeout_seconds)
        self.produce_retry_seconds = float(produce_retry_seconds)
        self.shutdown_timeout_seconds = float(shutdown_timeout_seconds)

        config = dict(producer_config)

        if not config.get("bootstrap.servers"):
            raise ValueError(
                "producer_config requires bootstrap.servers"
            )

        config.setdefault(
            "client.id",
            f"deepstream-{self.model_id}-protobuf-publisher",
        )

        self._producer = Producer(config)

        self._queue = queue.Queue(maxsize=queue_maxsize)

        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._thread = None
        self._started = False
        self._accepting = False
        self._closed = False
        self._fatal_error = None

        self._stats = {
            "model_id": self.model_id,
            "topic": self.topic,
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

    def start(self):
        with self._state_lock:

            if self._closed:
                raise RuntimeError("Publisher is already closed")

            if self._started:
                return self

            self._accepting = True
            self._started = True

            self._thread = threading.Thread(
                target=self._worker_loop,
                name=f"protobuf-kafka-{self.model_id}",
                daemon=False,
            )

            self._thread.start()

        return self

    def publish(
        self,
        message: InferenceResult,
        *,
        timeout: float | None = None,
    ):

        if not isinstance(message, InferenceResult):
            raise TypeError(
                "publish expects an InferenceResult Protobuf message"
            )

        if message.model_id != self.model_id:
            raise ValueError(
                f"{self.model_id} publisher cannot publish "
                f"{message.model_id}"
            )

        for field in ("camera_id", "frame_id", "result_id"):
            if not getattr(message, field):
                raise ValueError(f"Missing required Protobuf field: {field}")

        # Serialize to binary Protobuf, not JSON.
        payload = message.SerializeToString(deterministic=True)

        # All frames from one camera use the same Kafka partition key.
        key = message.camera_id.encode("utf-8")

        item = _QueuedMessage(
            payload=payload,
            key=key,
            frame_id=message.frame_id,
        )

        wait_seconds = (
            self.enqueue_timeout_seconds
            if timeout is None
            else timeout
        )

        if (
            isinstance(wait_seconds, bool)
            or not isinstance(wait_seconds, (int, float))
            or not math.isfinite(wait_seconds)
            or wait_seconds < 0
        ):
            raise ValueError("timeout must be non-negative and finite")

        with self._state_lock:

            if not self._started or not self._accepting:
                raise RuntimeError(
                    "Publisher is not accepting messages"
                )

            try:
                self._queue.put(
                    item,
                    timeout=float(wait_seconds),
                )

            except queue.Full as exc:
                self._increment("enqueue_timeouts")

                raise TimeoutError(
                    f"{self.model_id} queue remained full"
                ) from exc

        with self._stats_lock:
            self._stats["enqueued"] += 1
            self._stats["max_queue_depth"] = max(
                self._stats["max_queue_depth"],
                self._queue.qsize(),
            )

    def _increment(self, field, amount=1):
        with self._stats_lock:
            self._stats[field] += amount

    def _record_error(self, message):
        with self._stats_lock:
            errors = self._stats["last_errors"]
            errors.append(message)
            del errors[:-20]

    def _delivery_callback(self, error, kafka_message):

        if error is not None:
            self._increment("delivery_failures")
            self._record_error(f"Delivery failed: {error}")
        else:
            self._increment("delivered")

    def _produce_with_backpressure(self, item):

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

    def _submit_batch(self, batch):

        accepted = 0

        for item in batch:

            try:
                self._produce_with_backpressure(item)
                accepted += 1

            except Exception as exc:

                self._increment("publish_failures")

                self._record_error(
                    f"frame_id={item.frame_id}: "
                    f"{type(exc).__name__}: {exc}"
                )

            finally:
                self._queue.task_done()

        if accepted:
            self._increment("batches_submitted")

        self._producer.poll(0)

    def _worker_loop(self):

        try:

            while (
                not self._stop_event.is_set()
                or not self._queue.empty()
            ):

                try:
                    first = self._queue.get(
                        timeout=min(
                            self.batch_flush_interval_seconds,
                            0.1,
                        )
                    )

                except queue.Empty:
                    self._producer.poll(0)
                    continue

                batch = [first]

                deadline = (
                    time.monotonic()
                    + self.batch_flush_interval_seconds
                )

                while len(batch) < self.max_batch_messages:

                    remaining = deadline - time.monotonic()

                    if remaining <= 0:
                        break

                    try:
                        batch.append(
                            self._queue.get(timeout=remaining)
                        )

                    except queue.Empty:
                        break

                self._submit_batch(batch)

        except Exception as exc:

            self._increment("worker_errors")

            self._record_error(
                f"Worker failed: {type(exc).__name__}: {exc}"
            )

            self._fatal_error = (
                f"{type(exc).__name__}: {exc}"
            )

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
                pending = self._producer.flush(
                    self.shutdown_timeout_seconds
                )

                if pending:
                    self._increment("shutdown_pending", pending)

                    self._record_error(
                        f"{pending} Kafka message(s) remained pending"
                    )

            except Exception as exc:

                self._increment("worker_errors")

                self._record_error(
                    f"Flush failed: {type(exc).__name__}: {exc}"
                )

                self._fatal_error = self._fatal_error or str(exc)

    def close(self):

        with self._state_lock:

            if self._closed:
                return self.stats()

            self._accepting = False
            self._stop_event.set()

            thread = self._thread

            if not self._started:
                self._closed = True
                return self.stats()

        thread.join(
            timeout=self.shutdown_timeout_seconds + 5.0
        )

        if thread.is_alive():
            raise TimeoutError(
                f"{self.model_id} publisher did not stop"
            )

        with self._state_lock:
            self._closed = True

        result = self.stats()

        if self._fatal_error:
            raise RuntimeError(
                f"{self.model_id} publisher failed: "
                f"{self._fatal_error}"
            )

        return result

    def stats(self):

        with self._stats_lock:
            snapshot = dict(self._stats)
            snapshot["last_errors"] = list(
                self._stats["last_errors"]
            )

        snapshot["queue_depth"] = self._queue.qsize()

        snapshot["outstanding_after_produce"] = max(
            0,
            snapshot["produce_accepted"]
            - snapshot["delivered"]
            - snapshot["delivery_failures"],
        )

        return snapshot
