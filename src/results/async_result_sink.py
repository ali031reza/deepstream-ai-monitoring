"""Bounded asynchronous JSONL sink used as a local Kafka-publisher stand-in.

Create one sink per model. Each sink owns its queue, writer thread, and file.
A full queue applies backpressure rather than dropping inference results.
Batches are flushed when either the batch-size limit or flush interval is met.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from configs.paths import ProjectPaths
from typing import Any, Mapping

from src.results.model_result_message import serialize_result_message


_STOP = object()


class AsyncJSONLResultSink:
    """Write one model's result messages asynchronously to a JSONL file.

    The output file is opened in write mode for each run, so a new run starts
    with a clean file instead of silently appending to an older test run.
    """

    def __init__(
        self,
        *,
        model_id: str,
        output_path: str | Path,
        queue_size: int = 128,
        batch_size: int = 32,
        flush_interval_ms: int = 100,
    ) -> None:
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string.")
        if queue_size <= 0:
            raise ValueError("queue_size must be greater than zero.")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if flush_interval_ms <= 0:
            raise ValueError("flush_interval_ms must be greater than zero.")

        self.model_id = model_id
        self.output_path = ProjectPaths.from_environment().resolve(output_path)
        self.batch_size = int(batch_size)
        self.flush_interval_seconds = flush_interval_ms / 1000.0
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=int(queue_size))

        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._state_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._error_lock = threading.Lock()
        self._error: str | None = None

        self._stats = {
            "enqueued_messages": 0,
            "written_messages": 0,
            "batches_written": 0,
            "max_batch_size": 0,
            "max_queue_depth": 0,
            "blocked_submissions": 0,
            "submit_wait_total_ms": 0.0,
            "submit_wait_max_ms": 0.0,
        }

    @property
    def error(self) -> str | None:
        with self._error_lock:
            return self._error

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            snapshot = dict(self._stats)
        snapshot["model_id"] = self.model_id
        snapshot["output_path"] = str(self.output_path)
        snapshot["queue_depth"] = self._queue.qsize()
        snapshot["error"] = self.error
        return snapshot

    def start(self) -> None:
        with self._state_lock:
            if self._started:
                raise RuntimeError(f"Sink {self.model_id} has already been started.")
            if self._closed:
                raise RuntimeError(f"Sink {self.model_id} is already closed.")

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self._thread = threading.Thread(
                target=self._run,
                name=f"{self.model_id}-result-sink",
                daemon=False,
            )
            self._started = True
            self._thread.start()

    def submit(self, message: Mapping[str, Any]) -> None:
        """Queue a validated message; block on a full queue (no message drops)."""

        if message.get("model_id") != self.model_id:
            raise ValueError(
                f"Sink {self.model_id} cannot accept model_id "
                f"{message.get('model_id')!r}."
            )
        # Serialize once here to validate before accepting the message.
        # The writer serializes again when forming its batch.
        serialize_result_message(message)

        with self._state_lock:
            if not self._started or self._closed:
                raise RuntimeError(f"Sink {self.model_id} is not accepting messages.")

        submit_start = time.perf_counter_ns()
        blocked = False
        while True:
            error = self.error
            if error is not None:
                raise RuntimeError(f"Sink {self.model_id} failed: {error}")
            if not self.is_alive:
                raise RuntimeError(f"Sink {self.model_id} writer is not running.")
            try:
                self._queue.put(message, timeout=0.1)
                wait_ms = (time.perf_counter_ns() - submit_start) / 1_000_000.0
                with self._stats_lock:
                    self._stats["enqueued_messages"] += 1
                    self._stats["submit_wait_total_ms"] += wait_ms
                    self._stats["submit_wait_max_ms"] = max(
                        self._stats["submit_wait_max_ms"], wait_ms
                    )
                    self._stats["max_queue_depth"] = max(
                        self._stats["max_queue_depth"], self._queue.qsize()
                    )
                    if blocked:
                        self._stats["blocked_submissions"] += 1
                return
            except queue.Full:
                # Intentional bounded backpressure. Check writer health again.
                blocked = True
                continue

    def close(self) -> None:
        """Stop the writer after draining and flushing its remaining batch."""

        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread

        if thread is None:
            return

        while thread.is_alive():
            if self.error is not None:
                break
            try:
                self._queue.put(_STOP, timeout=0.1)
                break
            except queue.Full:
                continue

        thread.join()

    def _set_error(self, exc: BaseException) -> None:
        with self._error_lock:
            if self._error is None:
                self._error = f"{type(exc).__name__}: {exc}"

    def _run(self) -> None:
        try:
            with self.output_path.open("w", encoding="utf-8", newline="\n") as handle:
                stopping = False

                while not stopping:
                    try:
                        first = self._queue.get(
                            timeout=self.flush_interval_seconds
                        )
                    except queue.Empty:
                        continue

                    if first is _STOP:
                        self._queue.task_done()
                        break

                    batch = [first]
                    deadline = time.monotonic() + self.flush_interval_seconds

                    while len(batch) < self.batch_size:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        try:
                            item = self._queue.get(timeout=remaining)
                        except queue.Empty:
                            break

                        if item is _STOP:
                            self._queue.task_done()
                            stopping = True
                            break
                        batch.append(item)

                    try:
                        payload = "\n".join(
                            serialize_result_message(item) for item in batch
                        ) + "\n"
                        handle.write(payload)
                        handle.flush()

                        with self._stats_lock:
                            self._stats["written_messages"] += len(batch)
                            self._stats["batches_written"] += 1
                            self._stats["max_batch_size"] = max(
                                self._stats["max_batch_size"], len(batch)
                            )
                    finally:
                        for _ in batch:
                            self._queue.task_done()

                handle.flush()

        except BaseException as exc:
            self._set_error(exc)

        finally:
            # If the writer fails, release unfinished queue tasks so shutdown
            # cannot hang. submit() will observe the stored error and stop.
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._queue.task_done()
