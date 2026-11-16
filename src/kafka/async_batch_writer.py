from __future__ import annotations

import logging
import queue
import threading
import time

from dataclasses import dataclass
from typing import Callable


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchCompletion:
    records: tuple
    success: bool
    persisted_count: int
    dead_letter_count: int
    elapsed_ms: float
    error: str | None = None


class AsyncBatchWriter:
    """
    Single background writer. It never touches the Kafka consumer or commits
    offsets; it reports completed batches to the consumer thread.
    """

    def __init__(
        self,
        *,
        persist_batch: Callable,
        dead_letter_publisher,
        batch_size: int = 100,
        flush_interval_ms: int = 50,
        queue_capacity: int = 300,
        retry_attempts: int = 3,
        retry_backoff_ms: int = 250,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if flush_interval_ms <= 0:
            raise ValueError("flush_interval_ms must be positive")
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        if retry_attempts < 0:
            raise ValueError("retry_attempts cannot be negative")
        if retry_backoff_ms <= 0:
            raise ValueError("retry_backoff_ms must be positive")

        self.persist_batch = persist_batch
        self.dead_letter_publisher = dead_letter_publisher
        self.batch_size = batch_size
        self.flush_interval_seconds = flush_interval_ms / 1000.0
        self.retry_attempts = retry_attempts
        self.retry_backoff_seconds = retry_backoff_ms / 1000.0

        self._queue = queue.Queue(maxsize=queue_capacity)
        self._completions = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = None
        self._started = False
        self._fatal_error = None
        self._state_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = {
            "batches": 0,
            "messages": 0,
            "persisted": 0,
            "dead_lettered": 0,
            "failed_batches": 0,
            "max_batch_size": 0,
            "max_queue_depth": 0,
            "db_time_ms": 0.0,
            "dlt_time_ms": 0.0,
        }

    def start(self):
        with self._state_lock:
            if self._started:
                raise RuntimeError("Batch writer already started")
            self._started = True
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="postgres-async-batch-writer",
                daemon=False,
            )
            self._thread.start()
        return self

    def submit(self, record) -> bool:
        if not self._started:
            raise RuntimeError("Batch writer has not started")
        if self._fatal_error is not None:
            raise RuntimeError(
                f"Batch writer failed: {self._fatal_error}"
            )
        if self._stop_event.is_set():
            raise RuntimeError("Batch writer is shutting down")

        try:
            self._queue.put_nowait(record)
        except queue.Full:
            return False

        with self._stats_lock:
            self._stats["max_queue_depth"] = max(
                self._stats["max_queue_depth"],
                self._queue.qsize(),
            )
        return True

    @property
    def queue_depth(self):
        return self._queue.qsize()

    @property
    def queue_full(self):
        return self._queue.full()

    @property
    def fatal_error(self):
        return self._fatal_error

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def request_stop(self):
        self._stop_event.set()

    def join(self, timeout=None):
        if self._thread is None:
            return True
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def get_completion_nowait(self):
        try:
            return self._completions.get_nowait()
        except queue.Empty:
            return None

    def get_completion(self, timeout=None):
        return self._completions.get(timeout=timeout)

    def _retry(self, operation, operation_name):
        last_error = None
        for attempt in range(self.retry_attempts + 1):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                if attempt >= self.retry_attempts:
                    break
                delay = self.retry_backoff_seconds * (2 ** attempt)
                logger.warning(
                    "%s failed; retry %d/%d in %.2fs: %s",
                    operation_name,
                    attempt + 1,
                    self.retry_attempts,
                    delay,
                    exc,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"{operation_name} failed after "
            f"{self.retry_attempts + 1} attempt(s)"
        ) from last_error

    def _flush_batch(self, batch):
        if not batch:
            return True

        started = time.perf_counter()
        valid_records = [
            r for r in batch
            if r.result is not None and r.error_type is None
        ]
        invalid_records = [
            r for r in batch
            if r.result is None or r.error_type is not None
        ]

        try:
            db_started = time.perf_counter()
            if valid_records:
                self._retry(
                    lambda: self.persist_batch(
                        [record.result for record in valid_records]
                    ),
                    "PostgreSQL batch persistence",
                )
            db_elapsed = (time.perf_counter() - db_started) * 1000.0

            dlt_started = time.perf_counter()
            if invalid_records:
                self._retry(
                    lambda: self.dead_letter_publisher.publish_batch(
                        invalid_records
                    ),
                    "Dead-letter batch publication",
                )
            dlt_elapsed = (time.perf_counter() - dlt_started) * 1000.0

            elapsed = (time.perf_counter() - started) * 1000.0
            completion = BatchCompletion(
                records=tuple(batch),
                success=True,
                persisted_count=len(valid_records),
                dead_letter_count=len(invalid_records),
                elapsed_ms=elapsed,
            )

            with self._stats_lock:
                self._stats["batches"] += 1
                self._stats["messages"] += len(batch)
                self._stats["persisted"] += len(valid_records)
                self._stats["dead_lettered"] += len(invalid_records)
                self._stats["max_batch_size"] = max(
                    self._stats["max_batch_size"], len(batch)
                )
                self._stats["db_time_ms"] += db_elapsed
                self._stats["dlt_time_ms"] += dlt_elapsed

            logger.info(
                "Batch flushed: size=%d persisted=%d dlt=%d time=%.2fms",
                len(batch),
                len(valid_records),
                len(invalid_records),
                elapsed,
            )
        except Exception as exc:
            self._fatal_error = exc
            self._stop_event.set()
            with self._stats_lock:
                self._stats["failed_batches"] += 1

            elapsed = (time.perf_counter() - started) * 1000.0
            completion = BatchCompletion(
                records=tuple(batch),
                success=False,
                persisted_count=0,
                dead_letter_count=0,
                elapsed_ms=elapsed,
                error=f"{type(exc).__name__}: {exc}",
            )
            logger.exception("Fatal batch processing failure")
        finally:
            for _ in batch:
                self._queue.task_done()
            self._completions.put(completion)

        return completion.success

    def _discard_pending_queue(self):
        # These records are intentionally not acknowledged. Kafka redelivers
        # them when the consumer restarts.
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
            else:
                self._queue.task_done()

    def _worker_loop(self):
        batch = []
        deadline = None

        try:
            while True:
                if self._stop_event.is_set() and self._queue.empty():
                    if batch:
                        self._flush_batch(batch)
                        batch = []
                    break

                # Wake frequently enough to honor shutdown even with a long
                # configured flush interval. Do NOT flush prematurely on an
                # ordinary 100-ms queue timeout.
                timeout = min(
                    0.1,
                    max(0.0, deadline - time.monotonic())
                    if batch else 0.1,
                )

                try:
                    record = self._queue.get(timeout=timeout)
                except queue.Empty:
                    if batch and (
                        self._stop_event.is_set()
                        or time.monotonic() >= deadline
                    ):
                        success = self._flush_batch(batch)
                        batch = []
                        deadline = None
                        if not success:
                            self._discard_pending_queue()
                            break
                    continue

                batch.append(record)
                if len(batch) == 1:
                    deadline = (
                        time.monotonic() + self.flush_interval_seconds
                    )

                if len(batch) >= self.batch_size:
                    success = self._flush_batch(batch)
                    batch = []
                    deadline = None
                    if not success:
                        self._discard_pending_queue()
                        break

            logger.info("Async batch writer stopped")
        except Exception as exc:
            self._fatal_error = exc
            self._stop_event.set()
            logger.exception("Unexpected batch writer failure")

            if batch:
                completion = BatchCompletion(
                    records=tuple(batch),
                    success=False,
                    persisted_count=0,
                    dead_letter_count=0,
                    elapsed_ms=0.0,
                    error=str(exc),
                )
                for _ in batch:
                    self._queue.task_done()
                self._completions.put(completion)

            self._discard_pending_queue()

    def stats(self):
        with self._stats_lock:
            result = dict(self._stats)
        result["queue_depth"] = self.queue_depth
        return result
