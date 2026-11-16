"""DeepStream single-camera, two-model GPU frame fan-out with worker pools.

Each decoded frame is cloned once into owned GPU memory. The same CuPy
allocation is submitted once to each model's bounded queue. Each model queue
is serviced by an independently configurable number of worker threads; every
worker owns its own CUDA stream and model instance.

Results are converted to the repository's ModelResultMessage JSON envelope
and sent through one independent AsyncKafkaPublisher per model.

This is a lossless correctness-first test: full model queues apply bounded
backpressure. No task is intentionally dropped. A worker failure is surfaced;
if every worker in a model pool exits, queued tasks are drained and their
shared-frame reservations are released safely.
"""

from __future__ import annotations

import argparse
import math
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import cupy as cp

from pyservicemaker import BufferRetriever, Flow, Pipeline

from src.models.mock_gpu_model_a import MockGPUModelA
from src.models.mock_gpu_model_b import MockGPUModelB
from src.results.async_kafka_publisher import AsyncKafkaPublisher
from src.results.kafka_config import (
    load_kafka_config,
    producer_config_for_model,
    publisher_options,
    topic_for_model,
)
from src.results.model_result_message import ModelResultMessage


CUDA_DLPACK_DEVICE_TYPE = 2
CONSUMERS = ("model_a", "model_b")
MODEL_CLASSES = {
    "model_a": MockGPUModelA,
    "model_b": MockGPUModelB,
}
RESULT_MODEL_IDS = {
    "model_a": "mock_model_a",
    "model_b": "mock_model_b",
}


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value cannot be negative.")
    return number


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil((percent / 100.0) * len(ordered)) - 1
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


class FrameTask:
    """Queue-specific timing metadata for one model's frame task."""

    def __init__(self, packet: "SharedGPUFrame") -> None:
        self.packet = packet
        self._enqueued_ns: int | None = None
        self._enqueued_event = threading.Event()

    def mark_enqueued(self) -> None:
        self._enqueued_ns = time.perf_counter_ns()
        self._enqueued_event.set()

    def queue_wait_ms(self) -> float:
        if not self._enqueued_event.wait(timeout=30):
            raise TimeoutError("Queue task was not marked as enqueued.")
        assert self._enqueued_ns is not None
        return (time.perf_counter_ns() - self._enqueued_ns) / 1_000_000.0


class SharedGPUFrame:
    """One owned GPU frame shared by the two independent model queues."""

    def __init__(
        self,
        *,
        frame_id: int,
        deepstream_tensor: Any,
        gpu_array: cp.ndarray,
        on_release,
    ) -> None:
        self.frame_id = frame_id
        self._deepstream_tensor = deepstream_tensor
        self._gpu_array: cp.ndarray | None = gpu_array
        self.gpu_pointer = int(gpu_array.data.ptr)
        self.shape = tuple(gpu_array.shape)
        self.dtype = str(gpu_array.dtype)
        self.nbytes = int(gpu_array.nbytes)

        self._expected_consumers = set(CONSUMERS)
        self._released_consumers: set[str] = set()
        self._lock = threading.Lock()
        self._fully_released = False
        self._on_release = on_release

    def get_gpu_array(self) -> cp.ndarray:
        with self._lock:
            if self._fully_released or self._gpu_array is None:
                raise RuntimeError(
                    f"Frame {self.frame_id} has already been released."
                )
            return self._gpu_array

    def release(self, consumer: str) -> None:
        callback = None
        with self._lock:
            if consumer not in self._expected_consumers:
                raise ValueError(f"Unknown consumer: {consumer}")
            if consumer in self._released_consumers:
                raise RuntimeError(
                    f"{consumer} released frame {self.frame_id} more than once."
                )

            self._released_consumers.add(consumer)
            if self._released_consumers == self._expected_consumers:
                self._gpu_array = None
                self._deepstream_tensor = None
                self._fully_released = True
                callback = self._on_release

        if callback is not None:
            callback()


class ModelWorkerPool:
    """One model's bounded frame queue and configurable CUDA worker pool."""

    def __init__(
        self,
        *,
        model_id: str,
        gpu_id: int,
        worker_count: int,
        queue_size: int,
        model_config: dict[str, Any],
        source_id: str,
        publisher: AsyncKafkaPublisher,
        results: dict[str, list[dict[str, Any]]],
        results_lock: threading.Lock,
    ) -> None:
        if model_id not in CONSUMERS:
            raise ValueError(f"Unsupported model_id: {model_id}")
        if worker_count <= 0 or queue_size <= 0:
            raise ValueError("worker_count and queue_size must be positive")

        self.model_id = model_id
        self.result_model_id = RESULT_MODEL_IDS[model_id]
        self.gpu_id = gpu_id
        self.worker_count = worker_count
        self.queue: queue.Queue[FrameTask | None] = queue.Queue(maxsize=queue_size)
        self.model_config = dict(model_config)
        self.source_id = source_id
        self.publisher = publisher
        self.results = results
        self.results_lock = results_lock

        self._threads: list[threading.Thread] = []
        self._ready_events: dict[str, threading.Event] = {}
        self._state_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._started = False
        self._stopped = False
        self._failed = False
        # Reserve one active slot per configured worker before starting threads.
        self._active_workers = worker_count

        self._submitted_tasks = 0
        self._completed_inferences = 0
        self._failed_tasks = 0
        self._published_messages = 0
        self._publish_failures = 0
        self._dropped_tasks = 0
        self._max_queue_depth = 0

        self._worker_info: dict[str, dict[str, Any]] = {}
        self._worker_stats: dict[str, dict[str, Any]] = {}
        self._startup_errors: list[str] = []
        self._worker_errors: list[str] = []
        self._publish_errors: list[str] = []

        for index in range(worker_count):
            worker_id = f"{model_id}-worker-{index + 1:02d}"
            self._worker_stats[worker_id] = {
                "completed": 0,
                "failed": 0,
                "published": 0,
                "publish_failures": 0,
                "queue_wait_total_ms": 0.0,
                "processing_total_ms": 0.0,
            }

    def load_ratio(self) -> float:
        return self.queue.qsize() / self.queue.maxsize

    def start(self, startup_timeout: float = 90.0) -> None:
        with self._state_lock:
            if self._started:
                raise RuntimeError(f"{self.model_id} pool already started")
            self._started = True

        for index in range(self.worker_count):
            worker_id = f"{self.model_id}-worker-{index + 1:02d}"
            ready_event = threading.Event()
            self._ready_events[worker_id] = ready_event
            thread = threading.Thread(
                target=self._worker_main,
                name=worker_id,
                args=(worker_id, ready_event),
                daemon=False,
            )
            try:
                thread.start()
                self._threads.append(thread)
            except BaseException as exc:
                with self._state_lock:
                    self._active_workers -= 1
                    if self._active_workers == 0:
                        self._failed = True
                with self._stats_lock:
                    self._startup_errors.append(
                        f"{worker_id}: thread start failed: {type(exc).__name__}: {exc}"
                    )
                ready_event.set()

        deadline = time.monotonic() + startup_timeout
        for worker_id, ready_event in self._ready_events.items():
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not ready_event.wait(timeout=remaining):
                with self._stats_lock:
                    self._startup_errors.append(
                        f"{worker_id}: initialization timed out"
                    )

        with self._stats_lock:
            startup_errors = list(self._startup_errors)
        if startup_errors:
            raise RuntimeError(
                f"{self.model_id} worker startup failed: {startup_errors}"
            )

        with self._state_lock:
            if self._active_workers <= 0 or self._failed:
                raise RuntimeError(f"No healthy workers remain for {self.model_id}")

    def submit(self, task: FrameTask) -> None:
        """Enqueue without dropping; check pool health while applying backpressure."""
        while True:
            with self._state_lock:
                if self._failed or self._active_workers <= 0:
                    raise RuntimeError(f"No healthy workers remain for {self.model_id}")
                try:
                    # The short timeout lets a worker failure update pool state.
                    self.queue.put(task, timeout=0.05)
                except queue.Full:
                    pass
                else:
                    task.mark_enqueued()
                    with self._stats_lock:
                        self._submitted_tasks += 1
                        self._max_queue_depth = max(
                            self._max_queue_depth, self.queue.qsize()
                        )
                    return

    def _record_task_failure(self, worker_id: str, message: str) -> None:
        with self._stats_lock:
            self._failed_tasks += 1
            self._worker_stats[worker_id]["failed"] += 1
            self._worker_errors.append(f"{worker_id}: {message}")

    def _record_publish_failure(self, worker_id: str, message: str) -> None:
        with self._stats_lock:
            self._publish_failures += 1
            self._worker_stats[worker_id]["publish_failures"] += 1
            self._publish_errors.append(f"{worker_id}: {message}")

    def _worker_main(self, worker_id: str, ready_event: threading.Event) -> None:
        stream = None
        model = None
        initialized = False

        try:
            initialization_start = time.perf_counter()
            with cp.cuda.Device(self.gpu_id):
                stream = cp.cuda.Stream(non_blocking=True)
                model = MODEL_CLASSES[self.model_id](
                    cp=cp,
                    gpu_id=self.gpu_id,
                    stream=stream,
                    **self.model_config,
                )
                stream.synchronize()

            initialization_ms = (time.perf_counter() - initialization_start) * 1000.0
            with self._stats_lock:
                self._worker_info[worker_id] = {
                    "gpu_id": self.gpu_id,
                    "stream_pointer": int(stream.ptr),
                    "initialization_ms": initialization_ms,
                    "model_config": dict(self.model_config),
                    "sample_message": None,
                }

            initialized = True
            ready_event.set()
            print(
                f"[{worker_id}] ready; GPU={self.gpu_id}; "
                f"CUDA stream={stream.ptr}; init={initialization_ms:.2f} ms",
                flush=True,
            )

            while True:
                task = self.queue.get()
                try:
                    if task is None:
                        return
                    self._process_task(worker_id, task, model, stream)
                finally:
                    self.queue.task_done()

        except BaseException as exc:
            message = f"{type(exc).__name__}: {exc}"
            with self._stats_lock:
                if initialized:
                    self._worker_errors.append(f"{worker_id}: fatal worker error: {message}")
                else:
                    self._startup_errors.append(f"{worker_id}: {message}")
            traceback.print_exc()

        finally:
            ready_event.set()
            if stream is not None:
                try:
                    with cp.cuda.Device(self.gpu_id):
                        stream.synchronize()
                except Exception as exc:
                    with self._stats_lock:
                        self._worker_errors.append(
                            f"{worker_id}: final CUDA synchronization failed: {exc}"
                        )
            model = None
            stream = None

            should_drain = False
            with self._state_lock:
                self._active_workers -= 1
                if self._active_workers <= 0:
                    self._active_workers = 0
                    self._failed = True
                    should_drain = True

            if should_drain:
                self._drain_pending_tasks()

    def _process_task(
        self,
        worker_id: str,
        task: FrameTask,
        model: Any,
        stream: cp.cuda.Stream,
    ) -> None:
        packet = task.packet
        frame = None
        frame_pointer = None
        frame_object_id = None
        queue_wait_ms = 0.0
        processing_ms = 0.0
        inference_result = None
        inference_error = None
        release_error = None

        try:
            queue_wait_ms = task.queue_wait_ms()
            with cp.cuda.Device(self.gpu_id):
                frame = packet.get_gpu_array()
                frame_pointer = int(frame.data.ptr)
                frame_object_id = id(frame)

                processing_start = time.perf_counter_ns()
                inference_result = model.infer(frame, frame_id=packet.frame_id)
                # The input allocation must not be released until this worker's
                # CUDA stream has finished reading it.
                stream.synchronize()
                processing_ms = (
                    time.perf_counter_ns() - processing_start
                ) / 1_000_000.0

            if inference_result.get("model_id") != self.result_model_id:
                raise RuntimeError(
                    f"Unexpected result model_id {inference_result.get('model_id')!r}; "
                    f"expected {self.result_model_id!r}"
                )
            if int(inference_result.get("frame_id", -1)) != packet.frame_id:
                raise RuntimeError("Model result frame_id does not match input packet")

        except Exception as exc:
            inference_error = exc
            traceback.print_exc()

        finally:
            if frame is not None:
                try:
                    with cp.cuda.Device(self.gpu_id):
                        stream.synchronize()
                except Exception as exc:
                    inference_error = inference_error or exc
                del frame
                frame = None

            try:
                packet.release(self.model_id)
            except Exception as exc:
                release_error = exc

        if inference_error is not None:
            self._record_task_failure(
                worker_id,
                f"frame {packet.frame_id} inference failed: "
                f"{type(inference_error).__name__}: {inference_error}",
            )
            return

        if release_error is not None:
            self._record_task_failure(
                worker_id,
                f"frame {packet.frame_id} release failed: {release_error}",
            )
            return

        assert inference_result is not None
        with self._stats_lock:
            self._completed_inferences += 1
            worker_stats = self._worker_stats[worker_id]
            worker_stats["completed"] += 1
            worker_stats["queue_wait_total_ms"] += queue_wait_ms
            worker_stats["processing_total_ms"] += processing_ms

        record = {
            "frame_id": packet.frame_id,
            "model_id": self.model_id,
            "worker_id": worker_id,
            "gpu_id": self.gpu_id,
            "gpu_pointer": frame_pointer,
            "array_object_id": frame_object_id,
            "queue_wait_ms": queue_wait_ms,
            "processing_ms": processing_ms,
            "timings_ms": dict(inference_result.get("timings_ms", {})),
            "detection_count": int(inference_result.get("detection_count", 0)),
        }
        with self.results_lock:
            self.results[self.model_id].append(record)

        # Keep the repository's existing versioned JSON result envelope.
        # Worker timing is added to the existing timings map; the schema itself
        # is not changed in this phase.
        result_for_message = dict(inference_result)
        message_timings = dict(inference_result.get("timings_ms", {}))
        message_timings["worker_queue_wait_ms"] = round(queue_wait_ms, 5)
        message_timings["worker_processing_ms"] = round(processing_ms, 5)
        result_for_message["timings_ms"] = message_timings

        try:
            message = ModelResultMessage.from_inference_result(
                result_for_message,
                source_id=self.source_id,
            )
            self.publisher.publish(message)
            with self._stats_lock:
                self._published_messages += 1
                self._worker_stats[worker_id]["published"] += 1
                if self._worker_info[worker_id]["sample_message"] is None:
                    self._worker_info[worker_id]["sample_message"] = message.to_dict()
        except Exception as exc:
            self._record_publish_failure(
                worker_id,
                f"frame {packet.frame_id}: {type(exc).__name__}: {exc}",
            )
            traceback.print_exc()

    def _drain_pending_tasks(self) -> None:
        """Release queued frame reservations if every worker in this pool died."""
        while True:
            try:
                task = self.queue.get_nowait()
            except queue.Empty:
                break

            try:
                if task is not None:
                    try:
                        task.packet.release(self.model_id)
                    except Exception as exc:
                        with self._stats_lock:
                            self._worker_errors.append(
                                f"queue-drain release failed: {exc}"
                            )
                    with self._stats_lock:
                        self._dropped_tasks += 1
            finally:
                self.queue.task_done()

    def stop(self, timeout: float = 120.0) -> None:
        if self._stopped:
            return
        self._stopped = True
        if not self._threads:
            return

        # Wait for all accepted frame tasks to finish before placing sentinels.
        self.queue.join()
        live_threads = [thread for thread in self._threads if thread.is_alive()]
        for _ in live_threads:
            self.queue.put(None)

        deadline = time.monotonic() + timeout
        for thread in self._threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Timed out stopping {self.model_id} worker pool")
            thread.join(timeout=remaining)
            if thread.is_alive():
                raise TimeoutError(f"Worker {thread.name} did not stop")

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            worker_stats = {
                worker_id: dict(values)
                for worker_id, values in self._worker_stats.items()
            }
            worker_info = {
                worker_id: dict(values)
                for worker_id, values in self._worker_info.items()
            }
            startup_errors = list(self._startup_errors)
            worker_errors = list(self._worker_errors)
            publish_errors = list(self._publish_errors)
            snapshot = {
                "model_id": self.model_id,
                "configured_workers": self.worker_count,
                "started_workers": len(worker_info),
                "active_workers": self._active_workers,
                "submitted_tasks": self._submitted_tasks,
                "completed_inferences": self._completed_inferences,
                "failed_tasks": self._failed_tasks,
                "published_messages": self._published_messages,
                "publish_failures": self._publish_failures,
                "dropped_tasks": self._dropped_tasks,
                "max_queue_depth": self._max_queue_depth,
                "queue_depth": self.queue.qsize(),
                "worker_stats": worker_stats,
                "worker_info": worker_info,
                "startup_errors": startup_errors,
                "worker_errors": worker_errors,
                "publish_errors": publish_errors,
            }
        return snapshot


class FanoutRetriever(BufferRetriever):
    """Clone each DeepStream frame once and submit it once to each model pool."""

    def __init__(
        self,
        *,
        gpu_id: int,
        worker_pools: dict[str, ModelWorkerPool],
        sample_limit: int = 5,
    ) -> None:
        super().__init__()
        self.gpu_id = gpu_id
        self.worker_pools = worker_pools
        self.sample_limit = sample_limit
        self.frames_seen = 0
        self.frames_cloned = 0
        self.frames_submitted = 0
        self.frames_fully_released = 0
        self.errors = 0
        self._stats_lock = threading.Lock()

    def _on_frame_released(self) -> None:
        with self._stats_lock:
            self.frames_fully_released += 1

    def consume(self, buffer) -> int:
        with self._stats_lock:
            frame_id = self.frames_seen
            self.frames_seen += 1

        raw_tensor = None
        owned_tensor = None
        gpu_array = None
        packet = None
        submitted_to: set[str] = set()

        try:
            raw_tensor = buffer.extract(0)
            if raw_tensor is None:
                raise RuntimeError("Failed to extract DeepStream tensor")

            owned_tensor = raw_tensor.clone()
            del raw_tensor
            raw_tensor = None

            with cp.cuda.Device(self.gpu_id):
                device = tuple(owned_tensor.__dlpack_device__())
                expected_device = (CUDA_DLPACK_DEVICE_TYPE, self.gpu_id)
                if device != expected_device:
                    raise RuntimeError(
                        f"Unexpected DLPack device {device}; expected {expected_device}"
                    )

                gpu_array = cp.from_dlpack(owned_tensor)
                if gpu_array.device.id != self.gpu_id:
                    raise RuntimeError("CuPy array is on the wrong GPU")
                if gpu_array.dtype != cp.uint8:
                    raise RuntimeError(f"Unexpected frame dtype: {gpu_array.dtype}")
                if gpu_array.ndim != 3 or gpu_array.shape[2] != 3:
                    raise RuntimeError(
                        "Expected frame tensor with shape (H, W, 3); "
                        f"received {gpu_array.shape}"
                    )

                # Keep the synchronization barrier from the known-good fan-out
                # test until a separate event-based handoff experiment proves
                # that it can safely be removed.
                cp.cuda.runtime.deviceSynchronize()

            packet = SharedGPUFrame(
                frame_id=frame_id,
                deepstream_tensor=owned_tensor,
                gpu_array=gpu_array,
                on_release=self._on_frame_released,
            )
            owned_tensor = None
            gpu_array = None

            with self._stats_lock:
                self.frames_cloned += 1

            # Submit to the least-loaded pool first. If one model is saturated,
            # the other model gets this frame into its queue before upstream
            # backpressure waits for the saturated pool.
            ordered_models = sorted(
                CONSUMERS,
                key=lambda model_id: self.worker_pools[model_id].load_ratio(),
            )
            for model_id in ordered_models:
                task = FrameTask(packet)
                self.worker_pools[model_id].submit(task)
                submitted_to.add(model_id)

            with self._stats_lock:
                self.frames_submitted += 1

            if frame_id < self.sample_limit:
                print("\n========== SHARED GPU FRAME ==========")
                print(f"Frame ID: {packet.frame_id}")
                print(f"Shape: {packet.shape}")
                print(f"Dtype: {packet.dtype}")
                print(f"GPU ID: {self.gpu_id}")
                print(f"GPU pointer: {packet.gpu_pointer}")
                print(f"Frame memory: {packet.nbytes / 1024**2:.3f} MiB")
                print("Submitted once to Model A pool: SUCCESS")
                print("Submitted once to Model B pool: SUCCESS")
                print("======================================")

            if (frame_id + 1) % 100 == 0:
                print(f"[PRODUCER] Seen {frame_id + 1} frames", flush=True)
            return 1

        except Exception as exc:
            with self._stats_lock:
                self.errors += 1
            print(f"\n[PRODUCER ERROR] Frame {frame_id}: {exc}", flush=True)
            traceback.print_exc()

            if packet is not None:
                for model_id in CONSUMERS:
                    if model_id not in submitted_to:
                        try:
                            packet.release(model_id)
                        except Exception:
                            traceback.print_exc()
            return -1

        finally:
            if raw_tensor is not None:
                del raw_tensor
            if owned_tensor is not None:
                del owned_tensor
            if gpu_array is not None:
                del gpu_array

    def get_summary(self) -> dict[str, Any]:
        with self._stats_lock:
            return {
                "frames_seen": self.frames_seen,
                "frames_cloned": self.frames_cloned,
                "frames_submitted": self.frames_submitted,
                "frames_fully_released": self.frames_fully_released,
                "errors": self.errors,
            }


def print_model_statistics(
    model_id: str,
    records: list[dict[str, Any]],
    pool_stats: dict[str, Any],
) -> None:
    queue_waits = [item["queue_wait_ms"] for item in records]
    processing_times = [item["processing_ms"] for item in records]
    model_timings = [item["timings_ms"] for item in records]

    def average(key: str) -> float:
        values = [item.get(key, 0.0) for item in model_timings]
        return sum(values) / len(values) if values else 0.0

    print(f"\n========== {model_id.upper()} WORKER POOL ==========")
    print(f"Result model ID: {RESULT_MODEL_IDS[model_id]}")
    print(
        f"Workers configured / started: {pool_stats['configured_workers']} / "
        f"{pool_stats['started_workers']}"
    )
    print(f"Tasks submitted: {pool_stats['submitted_tasks']}")
    print(f"Inference completed: {pool_stats['completed_inferences']}")
    print(f"Inference failures: {pool_stats['failed_tasks']}")
    print(f"Messages accepted by publisher: {pool_stats['published_messages']}")
    print(f"Publish failures: {pool_stats['publish_failures']}")
    print(f"Dropped tasks: {pool_stats['dropped_tasks']}")
    print(f"Maximum input queue depth: {pool_stats['max_queue_depth']}")
    print(
        f"Queue wait average / P95: "
        f"{(sum(queue_waits) / len(queue_waits)) if queue_waits else 0.0:.3f} / "
        f"{percentile(queue_waits, 95):.3f} ms"
    )
    print(
        f"Worker processing average / P95: "
        f"{(sum(processing_times) / len(processing_times)) if processing_times else 0.0:.3f} / "
        f"{percentile(processing_times, 95):.3f} ms"
    )
    print(f"GPU preprocessing average: {average('preprocessing_gpu_ms'):.3f} ms")
    print(f"GPU inference average: {average('inference_gpu_ms'):.3f} ms")
    print(f"GPU postprocessing average: {average('postprocessing_gpu_ms'):.3f} ms")
    print(f"GPU total average: {average('gpu_total_ms'):.3f} ms")
    print(f"Model end-to-end average: {average('end_to_end_ms'):.3f} ms")

    print("\nPer-worker distribution:")
    for worker_id, worker_stats in pool_stats["worker_stats"].items():
        info = pool_stats["worker_info"].get(worker_id, {})
        print(
            f"  {worker_id}: completed={worker_stats['completed']}, "
            f"failed={worker_stats['failed']}, "
            f"published={worker_stats['published']}, "
            f"stream={info.get('stream_pointer', 'unavailable')}, "
            f"init={info.get('initialization_ms', 0.0):.2f} ms"
        )

    sample_messages = [
        info.get("sample_message")
        for info in pool_stats["worker_info"].values()
        if info.get("sample_message") is not None
    ]
    if sample_messages:
        print("Sample result envelope:")
        print(sample_messages[0])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "DeepStream shared-GPU-frame fan-out with independently "
            "configurable Model A and Model B worker pools and Kafka publishing"
        )
    )
    parser.add_argument("--source", required=True, help="Local MP4 file")
    parser.add_argument("--source-id", default="camera_0")
    parser.add_argument("--gpu-id", type=nonnegative_int, default=0)
    parser.add_argument("--sample-limit", type=nonnegative_int, default=5)
    parser.add_argument("--kafka-config", default=None)
    parser.add_argument("--worker-startup-timeout", type=positive_int, default=90)
    parser.add_argument("--worker-shutdown-timeout", type=positive_int, default=120)

    parser.add_argument("--model-a-workers", type=positive_int, default=1)
    parser.add_argument("--model-b-workers", type=positive_int, default=1)
    parser.add_argument("--queue-size", type=positive_int, default=8)
    parser.add_argument("--model-a-queue-size", type=positive_int, default=None)
    parser.add_argument("--model-b-queue-size", type=positive_int, default=None)

    parser.add_argument("--model-a-input-size", type=positive_int, default=320)
    parser.add_argument("--model-a-hidden-units", type=positive_int, default=128)
    parser.add_argument("--model-a-top-k", type=positive_int, default=5)
    parser.add_argument("--model-a-seed", type=int, default=42)

    parser.add_argument("--model-b-input-size", type=positive_int, default=320)
    parser.add_argument("--model-b-hidden-units", type=positive_int, default=512)
    parser.add_argument("--model-b-top-k", type=positive_int, default=5)
    parser.add_argument("--model-b-seed", type=int, default=84)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    video_path = Path(args.source)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not args.source_id.strip():
        raise ValueError("--source-id must not be empty")

    gpu_count = cp.cuda.runtime.getDeviceCount()
    if args.gpu_id >= gpu_count:
        raise RuntimeError(
            f"GPU {args.gpu_id} unavailable. Detected GPUs: {gpu_count}"
        )

    cp.cuda.Device(args.gpu_id).use()
    properties = cp.cuda.runtime.getDeviceProperties(args.gpu_id)
    gpu_name = properties["name"]
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()

    kafka_config = load_kafka_config(args.kafka_config)
    kafka_options = publisher_options(kafka_config)

    model_configs = {
        "model_a": {
            "input_size": args.model_a_input_size,
            "hidden_units": args.model_a_hidden_units,
            "top_k": args.model_a_top_k,
            "seed": args.model_a_seed,
        },
        "model_b": {
            "input_size": args.model_b_input_size,
            "hidden_units": args.model_b_hidden_units,
            "top_k": args.model_b_top_k,
            "seed": args.model_b_seed,
        },
    }
    worker_counts = {
        "model_a": args.model_a_workers,
        "model_b": args.model_b_workers,
    }
    queue_sizes = {
        "model_a": args.model_a_queue_size or args.queue_size,
        "model_b": args.model_b_queue_size or args.queue_size,
    }

    print("=" * 78)
    print("DEEPSTREAM TWO-MODEL GPU WORKER POOLS + KAFKA")
    print("=" * 78)
    print(f"Video: {video_path}")
    print(f"Source ID: {args.source_id}")
    print(f"GPU: {gpu_name} (ID {args.gpu_id})")
    print(f"Kafka bootstrap: {kafka_config['bootstrap_servers']}")
    for model_id in CONSUMERS:
        result_model_id = RESULT_MODEL_IDS[model_id]
        print(
            f"{model_id}: workers={worker_counts[model_id]}, "
            f"queue={queue_sizes[model_id]}, "
            f"topic={topic_for_model(kafka_config, result_model_id)}, "
            f"model_config={model_configs[model_id]}"
        )
    print("Frame transport: one owned GPU allocation shared by both model pools")
    print("Queue policy: bounded, blocking backpressure; no intentional drops")
    print("Result format: repository ModelResultMessage versioned JSON envelope")

    results: dict[str, list[dict[str, Any]]] = {model_id: [] for model_id in CONSUMERS}
    results_lock = threading.Lock()

    publishers: dict[str, AsyncKafkaPublisher] = {}
    pools: dict[str, ModelWorkerPool] = {}
    started_publishers: set[str] = set()
    publisher_stats: dict[str, dict[str, Any]] = {}
    publisher_errors: list[str] = []
    pipeline_error: BaseException | None = None
    pipeline_elapsed_seconds = 0.0
    retriever = None

    try:
        print("\nStarting independent Kafka publishers...", flush=True)
        for model_id in CONSUMERS:
            result_model_id = RESULT_MODEL_IDS[model_id]
            publisher = AsyncKafkaPublisher(
                model_id=result_model_id,
                topic=topic_for_model(kafka_config, result_model_id),
                producer_config=producer_config_for_model(
                    kafka_config, result_model_id
                ),
                **kafka_options,
            )
            publisher.start()
            publishers[model_id] = publisher
            started_publishers.add(model_id)

        print("Starting independent model worker pools...", flush=True)
        for model_id in CONSUMERS:
            pool = ModelWorkerPool(
                model_id=model_id,
                gpu_id=args.gpu_id,
                worker_count=worker_counts[model_id],
                queue_size=queue_sizes[model_id],
                model_config=model_configs[model_id],
                source_id=args.source_id,
                publisher=publishers[model_id],
                results=results,
                results_lock=results_lock,
            )
            pools[model_id] = pool
            pool.start(startup_timeout=float(args.worker_startup_timeout))

        print("\nAll model workers and Kafka publishers are ready.", flush=True)
        print("Building DeepStream pipeline...", flush=True)
        retriever = FanoutRetriever(
            gpu_id=args.gpu_id,
            worker_pools=pools,
            sample_limit=args.sample_limit,
        )

        pipeline_start = time.perf_counter()
        flow = (
            Flow(Pipeline("deepstream-two-model-worker-pools"))
            .capture([str(video_path)], gpu_id=args.gpu_id)
            .retrieve(retriever)
        )
        print("\nStarting video processing...\n", flush=True)
        flow()
        pipeline_elapsed_seconds = time.perf_counter() - pipeline_start

    except BaseException as exc:
        pipeline_error = exc
        if pipeline_elapsed_seconds == 0.0 and "pipeline_start" in locals():
            pipeline_elapsed_seconds = time.perf_counter() - pipeline_start
        print(f"\n[PIPELINE ERROR] {exc}", flush=True)
        traceback.print_exc()

    finally:
        print("\nStopping model worker pools and draining frame queues...", flush=True)
        for model_id in CONSUMERS:
            pool = pools.get(model_id)
            if pool is not None:
                try:
                    pool.stop(timeout=float(args.worker_shutdown_timeout))
                except Exception as exc:
                    publisher_errors.append(
                        f"{model_id} worker-pool shutdown: {type(exc).__name__}: {exc}"
                    )

        print("Closing Kafka publishers and flushing deliveries...", flush=True)
        for model_id in CONSUMERS:
            publisher = publishers.get(model_id)
            if publisher is not None and model_id in started_publishers:
                try:
                    publisher_stats[model_id] = publisher.close()
                except Exception as exc:
                    publisher_errors.append(
                        f"{model_id} publisher shutdown: {type(exc).__name__}: {exc}"
                    )
                    try:
                        publisher_stats[model_id] = publisher.stats()
                    except Exception:
                        pass

    if retriever is None:
        frame_summary = {
            "frames_seen": 0,
            "frames_cloned": 0,
            "frames_submitted": 0,
            "frames_fully_released": 0,
            "errors": 0,
        }
    else:
        frame_summary = retriever.get_summary()

    pool_stats = {
        model_id: pools[model_id].stats()
        for model_id in pools
    }

    print("\n" + "=" * 78)
    print("DEEPSTREAM + CONFIGURABLE GPU WORKER POOL RESULTS")
    print("=" * 78)
    print(f"Frames received: {frame_summary['frames_seen']}")
    print(f"GPU frame clones: {frame_summary['frames_cloned']}")
    print(f"Frames submitted to both models: {frame_summary['frames_submitted']}")
    print(f"Frames fully released: {frame_summary['frames_fully_released']}")
    print(f"Producer errors: {frame_summary['errors']}")
    print(f"Pipeline elapsed: {pipeline_elapsed_seconds:.3f} seconds")
    if pipeline_elapsed_seconds > 0:
        print(
            f"Submitted-frame throughput: "
            f"{frame_summary['frames_submitted'] / pipeline_elapsed_seconds:.2f} FPS"
        )

    for model_id in CONSUMERS:
        if model_id in pool_stats:
            print_model_statistics(model_id, results[model_id], pool_stats[model_id])

    print("\n========== KAFKA DELIVERY ==========")
    for model_id in CONSUMERS:
        stats = publisher_stats.get(model_id)
        if stats is None:
            print(f"{model_id}: publisher did not produce final statistics")
            continue
        print(
            f"{model_id}: topic={stats.get('topic')}, "
            f"enqueued={stats.get('enqueued')}, "
            f"delivered={stats.get('delivered')}, "
            f"delivery_failures={stats.get('delivery_failures')}, "
            f"publish_failures={stats.get('publish_failures')}, "
            f"outstanding={stats.get('outstanding_after_produce')}"
        )

    model_a_by_frame = {item["frame_id"]: item for item in results["model_a"]}
    model_b_by_frame = {item["frame_id"]: item for item in results["model_b"]}
    expected_frames = set(range(frame_summary["frames_submitted"]))
    matching_frames = (
        set(model_a_by_frame) == expected_frames
        and set(model_b_by_frame) == expected_frames
        and len(model_a_by_frame) == len(results["model_a"])
        and len(model_b_by_frame) == len(results["model_b"])
    )

    shared_pointer_matches = 0
    shared_object_matches = 0
    for frame_id in set(model_a_by_frame) & set(model_b_by_frame):
        result_a = model_a_by_frame[frame_id]
        result_b = model_b_by_frame[frame_id]
        if result_a["gpu_pointer"] == result_b["gpu_pointer"]:
            shared_pointer_matches += 1
        if result_a["array_object_id"] == result_b["array_object_id"]:
            shared_object_matches += 1

    stream_pointers = [
        info["stream_pointer"]
        for stats in pool_stats.values()
        for info in stats["worker_info"].values()
    ]
    expected_stream_count = sum(worker_counts.values())
    independent_streams = (
        len(stream_pointers) == expected_stream_count
        and len(set(stream_pointers)) == expected_stream_count
    )

    pool_checks = {}
    for model_id in CONSUMERS:
        stats = pool_stats.get(model_id, {})
        expected = frame_summary["frames_submitted"]
        pool_checks[model_id] = all([
            stats.get("submitted_tasks") == expected,
            stats.get("completed_inferences") == expected,
            stats.get("published_messages") == expected,
            stats.get("failed_tasks") == 0,
            stats.get("publish_failures") == 0,
            stats.get("dropped_tasks") == 0,
            not stats.get("startup_errors"),
            not stats.get("worker_errors"),
        ])

    kafka_checks = {}
    for model_id in CONSUMERS:
        stats = publisher_stats.get(model_id, {})
        expected = frame_summary["frames_submitted"]
        kafka_checks[model_id] = all([
            stats.get("enqueued") == expected,
            stats.get("delivered") == expected,
            stats.get("delivery_failures") == 0,
            stats.get("publish_failures") == 0,
            stats.get("outstanding_after_produce") == 0,
            stats.get("worker_errors") == 0,
        ])

    all_frames_released = (
        frame_summary["frames_fully_released"] == frame_summary["frames_cloned"]
    )
    shared_gpu_all_frames = (
        shared_pointer_matches == frame_summary["frames_submitted"]
        and shared_object_matches == frame_summary["frames_submitted"]
    )

    print("\n" + "=" * 78)
    print("ARCHITECTURE + WORKER POOL + KAFKA VALIDATION")
    print("=" * 78)
    print(f"Matching frame IDs in both model pools: {matching_frames}")
    print(
        f"Shared GPU pointer matches: {shared_pointer_matches}/"
        f"{frame_summary['frames_submitted']}"
    )
    print(
        f"Shared CuPy object matches: {shared_object_matches}/"
        f"{frame_summary['frames_submitted']}"
    )
    print(f"All cloned frames fully released: {all_frames_released}")
    print(f"All worker CUDA streams unique: {independent_streams}")
    for model_id in CONSUMERS:
        print(f"{model_id} worker-pool checks: {pool_checks.get(model_id, False)}")
        print(f"{model_id} Kafka checks: {kafka_checks.get(model_id, False)}")
    print(f"Producer errors: {frame_summary['errors']}")
    print(f"Shutdown errors: {len(publisher_errors)}")

    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    print("\n========== GPU MEMORY ==========")
    print(f"Total: {total_bytes / 1024**2:.2f} MiB")
    print(f"Free: {free_bytes / 1024**2:.2f} MiB")
    print(
        f"CuPy pool used: "
        f"{cp.get_default_memory_pool().used_bytes() / 1024**2:.2f} MiB"
    )

    successful = all([
        pipeline_error is None,
        frame_summary["errors"] == 0,
        frame_summary["frames_submitted"] > 0,
        frame_summary["frames_cloned"] == frame_summary["frames_submitted"],
        all_frames_released,
        matching_frames,
        shared_gpu_all_frames,
        independent_streams,
        all(pool_checks.values()),
        all(kafka_checks.values()),
        len(publisher_errors) == 0,
    ])

    print("\n" + "=" * 78)
    if successful:
        print("DEEPSTREAM TWO-MODEL WORKER POOL + KAFKA TEST PASSED")
    else:
        print("DEEPSTREAM TWO-MODEL WORKER POOL + KAFKA TEST FAILED")
        if pipeline_error is not None:
            print(f"PIPELINE ERROR: {pipeline_error}")
        for model_id, stats in pool_stats.items():
            for error in stats["startup_errors"] + stats["worker_errors"] + stats["publish_errors"]:
                print(f"{model_id} ERROR: {error}")
        for error in publisher_errors:
            print(f"SHUTDOWN ERROR: {error}")
    print("=" * 78)

    if not successful:
        raise RuntimeError("DeepStream two-model worker-pool/Kafka validation failed")


if __name__ == "__main__":
    main()
