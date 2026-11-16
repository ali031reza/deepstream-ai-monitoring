

"""
DeepStream shared GPU frame fan-out with independent binary Protobuf publishers.

Architecture:
    DeepStream -> one shared GPU frame -> Model A / Model B
    Model A -> Protobuf adapter -> AsyncProtobufKafkaPublisher
    Model B -> Protobuf adapter -> AsyncProtobufKafkaPublisher

Each model publishes its own InferenceResult message. No model aggregation.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
import traceback
from pathlib import Path
from uuid import uuid4

import cupy as cp
from confluent_kafka import Consumer, KafkaError, KafkaException
from google.protobuf.json_format import MessageToDict

from pyservicemaker import BufferRetriever, Flow, Pipeline

from src.models.mock_gpu_model_a import MockGPUModelA
from src.models.mock_gpu_model_b import MockGPUModelB

from src.proto.inference_result_pb2 import InferenceResult
from src.results.async_protobuf_kafka_publisher import AsyncProtobufKafkaPublisher
from src.results.kafka_config import (
    load_kafka_config,
    producer_config_for_model,
    publisher_options,
    topic_for_model,
)
from src.results.protobuf_result_adapter import build_inference_result


CUDA_DLPACK_DEVICE_TYPE = 2

CONSUMERS = ("model_a", "model_b")

MODEL_CLASSES = {
    "model_a": MockGPUModelA,
    "model_b": MockGPUModelB,
}

EXPECTED_RESULT_MODEL_IDS = {
    "model_a": "mock_model_a",
    "model_b": "mock_model_b",
}


# ============================================================
# ARGUMENT VALIDATION
# ============================================================

def positive_int(value):
    number = int(value)

    if number <= 0:
        raise argparse.ArgumentTypeError(
            "Value must be greater than zero."
        )

    return number


def nonnegative_int(value):
    number = int(value)

    if number < 0:
        raise argparse.ArgumentTypeError(
            "Value cannot be negative."
        )

    return number


def percentile(values, percent):
    if not values:
        return 0.0

    ordered = sorted(values)

    index = math.ceil(
        (percent / 100.0) * len(ordered)
    ) - 1

    index = max(
        0,
        min(index, len(ordered) - 1)
    )

    return ordered[index]


# ============================================================
# FRAME TASK
# ============================================================

class FrameTask:
    """
    Queue-specific metadata for one shared GPU frame.

    Each model gets its own FrameTask, but both tasks reference
    the same SharedGPUFrame object.
    """

    def __init__(self, packet):
        self.packet = packet

        self._enqueued_ns = None

        self._enqueued_event = threading.Event()

    def mark_enqueued(self):
        self._enqueued_ns = time.perf_counter_ns()

        self._enqueued_event.set()

    def queue_wait_ms(self):
        if not self._enqueued_event.wait(timeout=30):
            raise TimeoutError(
                "Queue task was not marked as enqueued."
            )

        return (
            time.perf_counter_ns() - self._enqueued_ns
        ) / 1_000_000.0


# ============================================================
# SHARED GPU FRAME
# ============================================================

class SharedGPUFrame:
    """
    One GPU allocation shared between Model A and Model B.

    The DeepStream tensor and CuPy view remain alive until both
    model workers release their ownership reservations.
    """

    def __init__(
        self,
        frame_id,
        captured_at_ms,
        deepstream_tensor,
        gpu_array,
        on_release,
    ):

        self.frame_id = frame_id
        self.captured_at_ms = int(captured_at_ms)

        self._deepstream_tensor = deepstream_tensor

        self._gpu_array = gpu_array

        self.gpu_pointer = int(
            gpu_array.data.ptr
        )

        self.shape = tuple(
            gpu_array.shape
        )

        self.dtype = str(
            gpu_array.dtype
        )

        self.nbytes = int(
            gpu_array.nbytes
        )

        self._expected_consumers = set(
            CONSUMERS
        )

        self._released_consumers = set()

        self._lock = threading.Lock()

        self._fully_released = False

        self._on_release = on_release

    def get_gpu_array(self):

        with self._lock:

            if (
                self._fully_released
                or self._gpu_array is None
            ):

                raise RuntimeError(
                    f"Frame {self.frame_id} "
                    "has already been released."
                )

            return self._gpu_array

    def release(self, consumer):

        callback = None

        with self._lock:

            if consumer not in self._expected_consumers:

                raise ValueError(
                    f"Unknown consumer: {consumer}"
                )

            if consumer in self._released_consumers:

                raise RuntimeError(
                    f"{consumer} released frame "
                    f"{self.frame_id} more than once."
                )

            self._released_consumers.add(
                consumer
            )

            if (
                self._released_consumers
                == self._expected_consumers
            ):

                self._gpu_array = None

                self._deepstream_tensor = None

                self._fully_released = True

                callback = self._on_release

        if callback is not None:
            callback()


# ============================================================
# DEEPSTREAM FRAME RETRIEVER
# ============================================================

class FanoutRetriever(BufferRetriever):
    """
    Extract each DeepStream frame once.

    Clone one GPU allocation and submit the same packet to
    two independent bounded model queues.
    """

    def __init__(
        self,
        gpu_id,
        queue_a,
        queue_b,
        sample_limit=5,
    ):

        super().__init__()

        self.gpu_id = gpu_id

        self.queues = {
            "model_a": queue_a,
            "model_b": queue_b,
        }

        self.sample_limit = sample_limit

        self.frames_seen = 0

        self.frames_cloned = 0

        self.frames_submitted = 0

        self.frames_fully_released = 0

        self.errors = 0

        self.max_queue_depth = {
            "model_a": 0,
            "model_b": 0,
        }

        self._stats_lock = threading.Lock()

    def _on_frame_released(self):

        with self._stats_lock:

            self.frames_fully_released += 1

    def consume(self, buffer):

        with self._stats_lock:

            frame_id = self.frames_seen

            self.frames_seen += 1

        # MP4 has no trustworthy wall-clock capture timestamp.
        # For this POC, record when DeepStream hands the frame to
        # the retriever; both model results reuse this same value.
        captured_at_ms = time.time_ns() // 1_000_000

        raw_tensor = None

        owned_tensor = None

        gpu_array = None

        packet = None

        submitted_to = set()

        try:

            # 1. Extract and clone the DeepStream tensor once.

            raw_tensor = buffer.extract(0)

            if raw_tensor is None:

                raise RuntimeError(
                    "Failed to extract DeepStream tensor."
                )

            owned_tensor = raw_tensor.clone()

            del raw_tensor

            raw_tensor = None

            # 2. Import the cloned allocation into CuPy.

            with cp.cuda.Device(self.gpu_id):

                device = tuple(
                    owned_tensor.__dlpack_device__()
                )

                expected_device = (
                    CUDA_DLPACK_DEVICE_TYPE,
                    self.gpu_id,
                )

                if device != expected_device:

                    raise RuntimeError(
                        f"Unexpected DLPack device {device}; "
                        f"expected {expected_device}."
                    )

                gpu_array = cp.from_dlpack(
                    owned_tensor
                )

                if gpu_array.device.id != self.gpu_id:

                    raise RuntimeError(
                        "CuPy array is on the wrong GPU."
                    )

                if gpu_array.dtype != cp.uint8:

                    raise RuntimeError(
                        f"Unexpected frame dtype: {gpu_array.dtype}"
                    )

                if (
                    gpu_array.ndim != 3
                    or gpu_array.shape[2] != 3
                ):

                    raise RuntimeError(
                        "Expected frame shape (H, W, 3); "
                        f"received {gpu_array.shape}."
                    )

                # Retain the previously validated readiness barrier.

                cp.cuda.runtime.deviceSynchronize()

            # 3. Transfer ownership to SharedGPUFrame.

            packet = SharedGPUFrame(
                frame_id=frame_id,
                captured_at_ms=captured_at_ms,
                deepstream_tensor=owned_tensor,
                gpu_array=gpu_array,
                on_release=self._on_frame_released,
            )

            owned_tensor = None

            gpu_array = None

            with self._stats_lock:

                self.frames_cloned += 1

            # 4. Submit the same frame to both queues.

            # Blocking puts preserve frames and apply backpressure.

            for consumer in CONSUMERS:

                task = FrameTask(packet)

                self.queues[consumer].put(task)

                submitted_to.add(consumer)

                task.mark_enqueued()

            with self._stats_lock:

                self.frames_submitted += 1

                for consumer in CONSUMERS:

                    self.max_queue_depth[consumer] = max(
                        self.max_queue_depth[consumer],
                        self.queues[consumer].qsize(),
                    )

            if frame_id < self.sample_limit:

                print("\n========== SHARED GPU FRAME ==========")

                print(f"Frame ID: {packet.frame_id}")

                print(f"Shape: {packet.shape}")

                print(f"Dtype: {packet.dtype}")

                print(f"GPU ID: {self.gpu_id}")

                print(f"GPU pointer: {packet.gpu_pointer}")

                print(
                    f"Frame memory: "
                    f"{packet.nbytes / 1024**2:.3f} MiB"
                )

                print("Submitted to Model A: SUCCESS")

                print("Submitted to Model B: SUCCESS")

                print("======================================")

            if (frame_id + 1) % 100 == 0:

                print(
                    f"[PRODUCER] Seen {frame_id + 1} frames",
                    flush=True,
                )

            return 1

        except Exception as exc:

            with self._stats_lock:

                self.errors += 1

            print(
                f"\n[PRODUCER ERROR] Frame {frame_id}: {exc}",
                flush=True,
            )

            traceback.print_exc()

            # Release reservations for queues that never
            # received this packet.

            if packet is not None:

                for consumer in CONSUMERS:

                    if consumer not in submitted_to:

                        try:

                            packet.release(consumer)

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

    def get_summary(self):

        with self._stats_lock:

            return {
                "frames_seen": self.frames_seen,
                "frames_cloned": self.frames_cloned,
                "frames_submitted": self.frames_submitted,
                "frames_fully_released": self.frames_fully_released,
                "errors": self.errors,
                "max_queue_depth_a": self.max_queue_depth["model_a"],
                "max_queue_depth_b": self.max_queue_depth["model_b"],
            }


# ============================================================
# ERROR MANAGEMENT
# ============================================================

def record_error(
    errors,
    errors_lock,
    model_id,
    message,
):

    with errors_lock:

        errors.append(
            f"{model_id}: {message}"
        )


def drain_queue(
    frame_queue,
    model_id,
    worker_errors,
    worker_errors_lock,
):
    """
    Drain pending frames after fatal worker failure.

    Each drained frame releases this worker's ownership
    reservation, allowing the other model to finish.
    """

    while True:

        task = frame_queue.get()

        try:

            if task is None:
                return

            try:

                task.packet.release(model_id)

            except Exception as exc:

                record_error(
                    worker_errors,
                    worker_errors_lock,
                    model_id,
                    f"Queue-drain release failed: {exc}",
                )

        finally:

            frame_queue.task_done()


# ============================================================
# GPU MODEL WORKER + KAFKA PUBLISHING
# ============================================================

def gpu_worker(
    model_id,
    frame_queue,
    gpu_id,
    model_config,
    source_id,
    source_session_id,
    kafka_publisher,
    results,
    results_lock,
    worker_errors,
    worker_errors_lock,
    startup_errors,
    startup_errors_lock,
    ready_event,
    worker_info,
):
    """
    Own one GPU model and one CUDA stream.

    Each worker adapts its own output to InferenceResult and
    publishes only to its model-specific Protobuf publisher.
    """

    stream = None

    model = None

    initialized = False

    fatal_error = None

    try:

        init_start = time.perf_counter()

        with cp.cuda.Device(gpu_id):

            stream = cp.cuda.Stream(
                non_blocking=True
            )

            model = MODEL_CLASSES[model_id](
                cp=cp,
                gpu_id=gpu_id,
                stream=stream,
                **model_config,
            )

            stream.synchronize()

        initialization_ms = (
            time.perf_counter() - init_start
        ) * 1000.0

        worker_info[model_id] = {
            "gpu_id": gpu_id,
            "stream_pointer": int(stream.ptr),
            "initialization_ms": initialization_ms,
            "model_config": dict(model_config),
            "sample_message": None,
        }

        initialized = True

        ready_event.set()

        print(
            f"[{model_id}] Initialized "
            f"{EXPECTED_RESULT_MODEL_IDS[model_id]} "
            f"on GPU {gpu_id}; "
            f"CUDA stream={stream.ptr}; "
            f"init={initialization_ms:.2f} ms",
            flush=True,
        )

        while True:

            task = frame_queue.get()

            try:

                if task is None:

                    print(
                        f"[{model_id}] Shutdown received.",
                        flush=True,
                    )

                    return

                packet = task.packet

                frame = None

                release_attempted = False

                publish_failed = False

                try:

                    queue_wait_ms = task.queue_wait_ms()

                    frame = packet.get_gpu_array()

                    frame_pointer = int(
                        frame.data.ptr
                    )

                    frame_object_id = id(frame)

                    processing_start = time.perf_counter_ns()

                    # GPU inference.

                    inference_result = model.infer(
                        frame,
                        frame_id=packet.frame_id,
                    )

                    # Wait for this model's CUDA stream only.

                    stream.synchronize()

                    processing_ms = (
                        time.perf_counter_ns()
                        - processing_start
                    ) / 1_000_000.0

                    # Inference is complete. Release this worker's
                    # frame reservation before Kafka backpressure.

                    del frame

                    frame = None

                    release_attempted = True

                    packet.release(model_id)

                    expected_result_id = (
                        EXPECTED_RESULT_MODEL_IDS[model_id]
                    )

                    if (
                        inference_result.get("model_id")
                        != expected_result_id
                    ):

                        raise RuntimeError(
                            f"Unexpected result model_id "
                            f"{inference_result.get('model_id')!r}; "
                            f"expected {expected_result_id!r}."
                        )

                    if (
                        int(inference_result.get("frame_id", -1))
                        != packet.frame_id
                    ):

                        raise RuntimeError(
                            "Model result frame_id does not "
                            "match input packet."
                        )

                    # Add pipeline-level timings to the model output.

                    result_for_message = dict(
                        inference_result
                    )

                    result_for_message["input_frame_shape"] = list(
                        packet.shape
                    )

                    timings = dict(
                        inference_result.get("timings_ms", {})
                    )

                    timings["queue_wait_ms"] = queue_wait_ms

                    timings["worker_processing_ms"] = processing_ms

                    result_for_message["timings_ms"] = timings

                    # Adapt directly to the backend InferenceResult contract.
                    # Event metadata remains false/empty until an event image
                    # has actually been stored by the event-frame stage.

                    result_message = build_inference_result(
                        result_for_message,
                        camera_id=source_id,
                        source_session_id=source_session_id,
                        frame_sequence=packet.frame_id,
                        captured_at_ms=packet.captured_at_ms,
                        worker_id=f"{model_id}-worker-01",
                    )

                    # Asynchronous enqueue to this model's publisher.
                    # A prolonged queue-full condition is fatal for this
                    # worker rather than silently dropping a result.

                    try:

                        kafka_publisher.publish(
                            result_message
                        )

                    except Exception:

                        publish_failed = True

                        raise

                    record = {
                        "frame_id": packet.frame_id,
                        "protobuf_frame_id": result_message.frame_id,
                        "result_id": result_message.result_id,
                        "captured_at_ms": result_message.captured_at_ms,
                        "processed_at_ms": result_message.processed_at_ms,
                        "model_id": model_id,
                        "gpu_id": gpu_id,
                        "gpu_pointer": frame_pointer,
                        "array_object_id": frame_object_id,
                        "queue_wait_ms": queue_wait_ms,
                        "processing_ms": processing_ms,
                        "timings_ms": dict(
                            inference_result.get("timings_ms", {})
                        ),
                        "detection_count": len(result_message.detections),
                    }

                    with results_lock:

                        results[model_id].append(record)

                    if (
                        worker_info[model_id]["sample_message"]
                        is None
                    ):

                        sample_message = MessageToDict(
                            result_message,
                            preserving_proto_field_name=True,
                        )
                        sample_message["detection_count"] = len(
                            result_message.detections
                        )
                        worker_info[model_id]["sample_message"] = (
                            sample_message
                        )

                except Exception as exc:

                    record_error(
                        worker_errors,
                        worker_errors_lock,
                        model_id,
                        f"Frame {packet.frame_id}: {exc}",
                    )

                    traceback.print_exc()

                    if publish_failed:

                        raise RuntimeError(
                            f"Kafka publishing failed for {model_id}; "
                            "stopping this worker to avoid silent "
                            "result loss."
                        ) from exc

                finally:

                    try:

                        stream.synchronize()

                    except Exception as exc:

                        record_error(
                            worker_errors,
                            worker_errors_lock,
                            model_id,
                            f"CUDA synchronization failed: {exc}",
                        )

                    if frame is not None:
                        del frame

                    if not release_attempted:

                        release_attempted = True

                        try:

                            packet.release(model_id)

                        except Exception as exc:

                            record_error(
                                worker_errors,
                                worker_errors_lock,
                                model_id,
                                f"Frame release failed: {exc}",
                            )

            finally:

                frame_queue.task_done()

    except BaseException as exc:

        fatal_error = exc

        if initialized:

            record_error(
                worker_errors,
                worker_errors_lock,
                model_id,
                f"Worker terminated unexpectedly: {exc}",
            )

        else:

            record_error(
                startup_errors,
                startup_errors_lock,
                model_id,
                str(exc),
            )

        traceback.print_exc()

    finally:

        ready_event.set()

        if fatal_error is not None:

            drain_queue(
                frame_queue,
                model_id,
                worker_errors,
                worker_errors_lock,
            )

        if stream is not None:

            try:

                stream.synchronize()

            except Exception as exc:

                record_error(
                    worker_errors,
                    worker_errors_lock,
                    model_id,
                    f"Final CUDA synchronization failed: {exc}",
                )

        model = None

        stream = None


# ============================================================
# STATISTICS
# ============================================================

def print_worker_statistics(
    model_id,
    records,
    worker_info,
):

    queue_waits = [
        item["queue_wait_ms"]
        for item in records
    ]

    processing_times = [
        item["processing_ms"]
        for item in records
    ]

    model_timings = [
        item["timings_ms"]
        for item in records
    ]

    def average(key):

        values = [
            item.get(key, 0.0)
            for item in model_timings
        ]

        return (
            sum(values) / len(values)
            if values else 0.0
        )

    print(f"\n========== {model_id.upper()} ==========")

    print(
        f"Model result ID: "
        f"{EXPECTED_RESULT_MODEL_IDS[model_id]}"
    )

    print(
        f"Completed inference results: {len(records)}"
    )

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

    print(
        f"GPU preprocessing average: "
        f"{average('preprocessing_gpu_ms'):.3f} ms"
    )

    print(
        f"GPU inference average: "
        f"{average('inference_gpu_ms'):.3f} ms"
    )

    print(
        f"GPU postprocessing average: "
        f"{average('postprocessing_gpu_ms'):.3f} ms"
    )

    print(
        f"GPU total average: "
        f"{average('gpu_total_ms'):.3f} ms"
    )

    info = worker_info.get(model_id, {})

    print(
        f"CUDA stream pointer: "
        f"{info.get('stream_pointer', 'unavailable')}"
    )

    print(
        f"Model initialization: "
        f"{info.get('initialization_ms', 0.0):.3f} ms"
    )

    sample = info.get("sample_message")

    if sample is not None:

        print(
            f"Sample detection count: "
            f"{sample['detection_count']}"
        )

        print("Sample Kafka message:")

        print(
            json.dumps(sample, indent=2)
        )

    print("================================")


def print_publisher_statistics(
    model_id,
    stats,
):

    print(f"\n========== KAFKA {model_id.upper()} ==========")

    for key in (
        "model_id",
        "topic",
        "enqueued",
        "produce_accepted",
        "delivered",
        "delivery_failures",
        "publish_failures",
        "enqueue_timeouts",
        "batches_submitted",
        "max_queue_depth",
        "queue_depth",
        "worker_errors",
        "shutdown_pending",
        "outstanding_after_produce",
    ):

        print(f"{key}: {stats.get(key)}")

    if stats.get("last_errors"):

        print("Recent Kafka errors:")

        for error in stats["last_errors"]:
            print(f"  - {error}")

    print("================================")


def wait_for_consumer_assignment(
    consumer,
    timeout_seconds=30.0,
):
    """Wait until the verification consumer owns its result-topic partitions."""

    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:

        assignment = consumer.assignment()

        if assignment:
            return assignment

        message = consumer.poll(0.1)

        if message is not None and message.error():

            if message.error().code() != KafkaError._PARTITION_EOF:
                raise KafkaException(message.error())

    raise TimeoutError(
        "Kafka verification consumer did not receive a partition assignment."
    )


def verify_protobuf_kafka_results(
    consumer,
    *,
    topic_to_model,
    camera_id,
    source_session_id,
    expected_frames,
    timeout_seconds=60.0,
):
    """
    Consume and validate the actual binary Protobuf messages emitted by
    this DeepStream run. Messages from other source sessions are ignored.
    """

    received = {
        model_id: {}
        for model_id in CONSUMERS
    }

    expected_total = expected_frames * len(CONSUMERS)
    deadline = time.monotonic() + timeout_seconds

    while sum(len(items) for items in received.values()) < expected_total:

        remaining = deadline - time.monotonic()

        if remaining <= 0:
            break

        kafka_message = consumer.poll(min(1.0, remaining))

        if kafka_message is None:
            continue

        if kafka_message.error():

            if kafka_message.error().code() == KafkaError._PARTITION_EOF:
                continue

            raise KafkaException(kafka_message.error())

        model_key = topic_to_model.get(kafka_message.topic())

        if model_key is None:
            continue

        payload = kafka_message.value()

        if not payload:
            continue

        decoded = InferenceResult()
        decoded.ParseFromString(payload)

        # Production topics may contain messages from other cameras/runs.
        if decoded.source_session_id != source_session_id:
            continue

        expected_model_id = EXPECTED_RESULT_MODEL_IDS[model_key]

        if decoded.model_id != expected_model_id:
            raise AssertionError(
                f"{model_key}: Protobuf model_id mismatch: "
                f"{decoded.model_id!r}"
            )

        if decoded.camera_id != camera_id:
            raise AssertionError(
                f"{model_key}: camera_id mismatch"
            )

        if decoded.schema_version != "1.0":
            raise AssertionError(
                f"{model_key}: schema version mismatch"
            )

        if decoded.status != "processed":
            raise AssertionError(
                f"{model_key}: unexpected result status {decoded.status!r}"
            )

        sequence = int(decoded.frame_sequence)

        if not 0 <= sequence < expected_frames:
            raise AssertionError(
                f"{model_key}: frame sequence out of range: {sequence}"
            )

        expected_frame_id = f"{source_session_id}:{sequence:012d}"

        if decoded.frame_id != expected_frame_id:
            raise AssertionError(
                f"{model_key}: frame_id does not match source session/sequence"
            )

        if not decoded.result_id:
            raise AssertionError(
                f"{model_key}: missing deterministic result_id"
            )

        if decoded.captured_at_ms <= 0 or decoded.processed_at_ms <= 0:
            raise AssertionError(
                f"{model_key}: missing capture/processing timestamp"
            )

        if not decoded.detections:
            raise AssertionError(
                f"{model_key}: no detections were serialized"
            )

        timing_names = {
            timing.stage
            for timing in decoded.timings
        }

        if not {"queue_wait_ms", "worker_processing_ms"}.issubset(
            timing_names
        ):
            raise AssertionError(
                f"{model_key}: pipeline timing metadata is missing"
            )

        if sequence in received[model_key]:
            raise AssertionError(
                f"{model_key}: duplicate Protobuf result for frame {sequence}"
            )

        received[model_key][sequence] = decoded

    expected_sequences = set(range(expected_frames))

    for model_key in CONSUMERS:

        actual_sequences = set(received[model_key])

        if actual_sequences != expected_sequences:
            missing = sorted(expected_sequences - actual_sequences)
            raise AssertionError(
                f"{model_key}: received {len(actual_sequences)}/"
                f"{expected_frames} Protobuf results; missing={missing[:10]}"
            )

    # Both models must refer to the same source frame and capture timestamp.
    for sequence in range(expected_frames):

        result_a = received["model_a"][sequence]
        result_b = received["model_b"][sequence]

        if result_a.frame_id != result_b.frame_id:
            raise AssertionError(
                f"Frame {sequence}: model frame identities differ"
            )

        if result_a.captured_at_ms != result_b.captured_at_ms:
            raise AssertionError(
                f"Frame {sequence}: capture timestamps differ"
            )

        if result_a.result_id == result_b.result_id:
            raise AssertionError(
                f"Frame {sequence}: model result IDs are not independent"
            )

    return {
        "passed": True,
        "messages_received": expected_total,
        "per_model": {
            model_key: len(received[model_key])
            for model_key in CONSUMERS
        },
    }


# ============================================================
# ARGUMENT PARSER
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "DeepStream shared GPU frame fan-out with "
            "independent asynchronous Kafka publishers."
        )
    )

    parser.add_argument(
        "--source",
        required=True,
    )

    parser.add_argument(
        "--source-id",
        default="camera_0",
    )

    parser.add_argument(
        "--gpu-id",
        type=nonnegative_int,
        default=0,
    )

    parser.add_argument(
        "--queue-size",
        type=positive_int,
        default=4,
    )

    parser.add_argument(
        "--sample-limit",
        type=nonnegative_int,
        default=5,
    )

    parser.add_argument(
        "--kafka-config",
        default=None,
        help="Path to config/kafka.json.",
    )

    # Optional overrides. If omitted, config/kafka.json is used.

    parser.add_argument(
        "--result-queue-size",
        type=positive_int,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=positive_int,
        default=None,
    )

    parser.add_argument(
        "--flush-interval-ms",
        type=positive_int,
        default=None,
    )

    parser.add_argument(
        "--model-a-input-size",
        type=positive_int,
        default=320,
    )

    parser.add_argument(
        "--model-a-hidden-units",
        type=positive_int,
        default=128,
    )

    parser.add_argument(
        "--model-a-top-k",
        type=positive_int,
        default=5,
    )

    parser.add_argument(
        "--model-a-seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--model-b-input-size",
        type=positive_int,
        default=320,
    )

    parser.add_argument(
        "--model-b-hidden-units",
        type=positive_int,
        default=512,
    )

    parser.add_argument(
        "--model-b-top-k",
        type=positive_int,
        default=5,
    )

    parser.add_argument(
        "--model-b-seed",
        type=int,
        default=84,
    )

    return parser


# ============================================================
# MAIN
# ============================================================

def main():

    args = build_parser().parse_args()

    video_path = Path(args.source)

    if not video_path.is_file():

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    if not args.source_id.strip():

        raise ValueError(
            "--source-id must not be empty."
        )

    # Every application run gets a distinct identity, even if the
    # source frame counter restarts at zero.
    source_session_id = uuid4().hex

    # --------------------------------------------------------
    # GPU INITIALIZATION
    # --------------------------------------------------------

    gpu_count = cp.cuda.runtime.getDeviceCount()

    if args.gpu_id >= gpu_count:

        raise RuntimeError(
            f"GPU {args.gpu_id} unavailable. "
            f"Detected GPUs: {gpu_count}"
        )

    cp.cuda.Device(args.gpu_id).use()

    properties = cp.cuda.runtime.getDeviceProperties(
        args.gpu_id
    )

    gpu_name = properties["name"]

    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()

    # --------------------------------------------------------
    # KAFKA CONFIGURATION
    # --------------------------------------------------------

    kafka_config = load_kafka_config(
        args.kafka_config
    )

    topics = {
        model_id: topic_for_model(
            kafka_config,
            EXPECTED_RESULT_MODEL_IDS[model_id],
        )
        for model_id in CONSUMERS
    }

    if len(set(topics.values())) != len(CONSUMERS):

        raise ValueError(
            "Each model must have its own independent Kafka topic."
        )

    options = publisher_options(
        kafka_config
    )

    if args.result_queue_size is not None:

        options["queue_maxsize"] = args.result_queue_size

    if args.batch_size is not None:

        options["max_batch_messages"] = args.batch_size

    if args.flush_interval_ms is not None:

        options["batch_flush_interval_ms"] = (
            args.flush_interval_ms
        )

    # One independent Kafka publisher per AI model.

    publishers = {

        model_id: AsyncProtobufKafkaPublisher(
            model_id=EXPECTED_RESULT_MODEL_IDS[model_id],
            topic=topics[model_id],
            producer_config=producer_config_for_model(
                kafka_config,
                EXPECTED_RESULT_MODEL_IDS[model_id],
            ),
            **options,
        )

        for model_id in CONSUMERS
    }

    consumer = Consumer({
        "bootstrap.servers": kafka_config["bootstrap_servers"],
        "group.id": f"deepstream-protobuf-verify-{source_session_id}",
        "auto.offset.reset": "latest",
        "enable.auto.commit": False,
    })

    # --------------------------------------------------------
    # MODEL CONFIGURATION
    # --------------------------------------------------------

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

    print("=" * 72)

    print(
        "DEEPSTREAM + SHARED GPU FRAME FAN-OUT "
        "+ TWO INDEPENDENT BINARY PROTOBUF PUBLISHERS"
    )

    print("=" * 72)

    print(f"Video: {video_path}")

    print(f"Camera ID: {args.source_id}")

    print(f"Source session ID: {source_session_id}")

    print(
        "Capture timestamp mode: DeepStream receiver wall-clock "
        "(MP4 proof-of-concept)"
    )

    print(f"GPU: {gpu_name}")

    print(f"GPU ID: {args.gpu_id}")

    print(
        f"Model A topic: {topics['model_a']}"
    )

    print(
        f"Model B topic: {topics['model_b']}"
    )

    print(
        f"Input queue capacity per model: {args.queue_size}"
    )

    print(
        f"Kafka queue capacity per model: "
        f"{options['queue_maxsize']}"
    )

    print(
        f"Kafka batch size: "
        f"{options['max_batch_messages']}"
    )

    print(
        f"Kafka batch interval: "
        f"{options['batch_flush_interval_ms']} ms"
    )

    print(
        "Publishing mode: independent asynchronous Kafka producers"
    )

    # --------------------------------------------------------
    # QUEUES AND STATISTICS
    # --------------------------------------------------------

    queues = {

        model_id: queue.Queue(
            maxsize=args.queue_size
        )

        for model_id in CONSUMERS
    }

    results = {
        "model_a": [],
        "model_b": [],
    }

    results_lock = threading.Lock()

    worker_errors = []

    worker_errors_lock = threading.Lock()

    startup_errors = []

    startup_errors_lock = threading.Lock()

    worker_info = {}

    ready_events = {

        model_id: threading.Event()

        for model_id in CONSUMERS
    }

    # --------------------------------------------------------
    # WORKER THREADS
    # --------------------------------------------------------

    threads = {

        model_id: threading.Thread(
            target=gpu_worker,
            name=f"{model_id.replace('_', '-')}-worker",
            args=(
                model_id,
                queues[model_id],
                args.gpu_id,
                model_configs[model_id],
                args.source_id,
                source_session_id,
                publishers[model_id],
                results,
                results_lock,
                worker_errors,
                worker_errors_lock,
                startup_errors,
                startup_errors_lock,
                ready_events[model_id],
                worker_info,
            ),
            daemon=False,
        )

        for model_id in CONSUMERS
    }

    # --------------------------------------------------------
    # DEEPSTREAM RETRIEVER
    # --------------------------------------------------------

    retriever = FanoutRetriever(
        gpu_id=args.gpu_id,
        queue_a=queues["model_a"],
        queue_b=queues["model_b"],
        sample_limit=args.sample_limit,
    )

    started_publishers = set()

    started_workers = set()

    publisher_stats = {}

    pipeline_error = None

    pipeline_elapsed_seconds = 0.0

    # --------------------------------------------------------
    # START EVERYTHING
    # --------------------------------------------------------

    try:

        print(
            "\nStarting Kafka Protobuf verification consumer...",
            flush=True,
        )

        consumer.subscribe(list(topics.values()))

        assignment = wait_for_consumer_assignment(
            consumer,
            timeout_seconds=30.0,
        )

        print(
            f"Kafka verifier assigned {len(assignment)} partition(s).",
            flush=True,
        )

        print(
            "\nStarting independent Protobuf Kafka publishers...",
            flush=True,
        )

        for model_id in CONSUMERS:

            publishers[model_id].start()

            started_publishers.add(model_id)

            print(
                f"[{model_id}] Kafka publisher started "
                f"for topic {topics[model_id]}",
                flush=True,
            )

        print(
            "\nStarting independent GPU workers...",
            flush=True,
        )

        for model_id in CONSUMERS:

            threads[model_id].start()

            started_workers.add(model_id)

        # Wait until both GPU models initialize.

        for model_id in CONSUMERS:

            if not ready_events[model_id].wait(timeout=60):

                raise TimeoutError(
                    f"{model_id} initialization "
                    "did not finish within 60 seconds."
                )

        if startup_errors:

            raise RuntimeError(
                f"GPU worker initialization failed: {startup_errors}"
            )

        print(
            "\nBoth GPU workers and Kafka publishers are ready.",
            flush=True,
        )

        # ----------------------------------------------------
        # START DEEPSTREAM
        # ----------------------------------------------------

        print(
            "\nBuilding DeepStream pipeline...",
            flush=True,
        )

        pipeline_start = time.perf_counter()

        flow = (

            Flow(
                Pipeline(
                    "shared-gpu-frame-kafka-publishing"
                )
            )

            .capture(
                [str(video_path)],
                gpu_id=args.gpu_id,
            )

            .retrieve(retriever)
        )

        print(
            "\nStarting video processing...\n",
            flush=True,
        )

        flow()

        pipeline_elapsed_seconds = (
            time.perf_counter() - pipeline_start
        )

    except BaseException as exc:

        pipeline_error = exc

        if (
            pipeline_elapsed_seconds == 0.0
            and "pipeline_start" in locals()
        ):

            pipeline_elapsed_seconds = (
                time.perf_counter() - pipeline_start
            )

        print(
            f"\n[PIPELINE ERROR] {exc}",
            flush=True,
        )

        traceback.print_exc()

    finally:

        # ----------------------------------------------------
        # STOP MODEL WORKERS FIRST
        # ----------------------------------------------------

        print(
            "\nSending model-worker shutdown signals...",
            flush=True,
        )

        for model_id in started_workers:

            queues[model_id].put(None)

        for model_id in started_workers:

            queues[model_id].join()

        for model_id in started_workers:

            threads[model_id].join()

        print(
            "All model workers stopped.",
            flush=True,
        )

        # ----------------------------------------------------
        # FLUSH AND CLOSE KAFKA PUBLISHERS
        # ----------------------------------------------------

        print(
            "\nClosing Kafka publishers and flushing pending messages...",
            flush=True,
        )

        for model_id in started_publishers:

            try:

                publisher_stats[model_id] = (
                    publishers[model_id].close()
                )

            except Exception as exc:

                worker_errors.append(
                    f"{model_id} Kafka shutdown: "
                    f"{type(exc).__name__}: {exc}"
                )

                print(
                    f"[KAFKA SHUTDOWN ERROR] "
                    f"{model_id}: {exc}",
                    flush=True,
                )

                publisher_stats[model_id] = (
                    publishers[model_id].stats()
                )

        # Capture statistics even if a publisher never started.

        for model_id in CONSUMERS:

            if model_id not in publisher_stats:

                publisher_stats[model_id] = (
                    publishers[model_id].stats()
                )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    summary = retriever.get_summary()

    kafka_verification = {
        "passed": False,
        "messages_received": 0,
        "per_model": {},
        "error": None,
    }

    try:

        if pipeline_error is None and summary["frames_submitted"] > 0:

            kafka_verification = verify_protobuf_kafka_results(
                consumer,
                topic_to_model={
                    topics[model_id]: model_id
                    for model_id in CONSUMERS
                },
                camera_id=args.source_id,
                source_session_id=source_session_id,
                expected_frames=summary["frames_submitted"],
                timeout_seconds=60.0,
            )

        else:

            kafka_verification["error"] = (
                "Pipeline did not complete a non-empty frame run."
            )

    except Exception as exc:

        kafka_verification["error"] = (
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

    finally:

        consumer.close()

    print("\n" + "=" * 72)

    print(
        "DEEPSTREAM + GPU INFERENCE + BINARY PROTOBUF KAFKA RESULTS"
    )

    print("=" * 72)

    print(
        f"Frames received: {summary['frames_seen']}"
    )

    print(
        f"GPU frame clones created: {summary['frames_cloned']}"
    )

    print(
        f"Frames submitted to both models: "
        f"{summary['frames_submitted']}"
    )

    print(
        f"Frames fully released: "
        f"{summary['frames_fully_released']}"
    )

    print(
        f"Producer errors: {summary['errors']}"
    )

    print(
        f"Model A completed: {len(results['model_a'])}"
    )

    print(
        f"Model B completed: {len(results['model_b'])}"
    )

    print(
        f"Pipeline elapsed: {pipeline_elapsed_seconds:.3f} seconds"
    )

    throughput = (

        summary["frames_submitted"] / pipeline_elapsed_seconds

        if pipeline_elapsed_seconds > 0

        else 0.0
    )

    print(
        f"Submitted-frame throughput: {throughput:.2f} FPS"
    )

    print(
        f"Model A maximum input queue depth: "
        f"{summary['max_queue_depth_a']}"
    )

    print(
        f"Model B maximum input queue depth: "
        f"{summary['max_queue_depth_b']}"
    )

    # --------------------------------------------------------
    # MODEL STATISTICS
    # --------------------------------------------------------

    for model_id in CONSUMERS:

        print_worker_statistics(
            model_id,
            results[model_id],
            worker_info,
        )

    # --------------------------------------------------------
    # KAFKA STATISTICS
    # --------------------------------------------------------

    for model_id in CONSUMERS:

        print_publisher_statistics(
            model_id,
            publisher_stats[model_id],
        )

    # --------------------------------------------------------
    # FRAME VALIDATION
    # --------------------------------------------------------

    model_a_by_frame = {

        item["frame_id"]: item

        for item in results["model_a"]
    }

    model_b_by_frame = {

        item["frame_id"]: item

        for item in results["model_b"]
    }

    expected_frames = set(
        range(summary["frames_submitted"])
    )

    model_a_frames = set(
        model_a_by_frame
    )

    model_b_frames = set(
        model_b_by_frame
    )

    matching_frames = (

        model_a_frames == expected_frames

        and model_b_frames == expected_frames
    )

    shared_pointer_matches = 0

    shared_object_matches = 0

    for frame_id in model_a_frames & model_b_frames:

        result_a = model_a_by_frame[frame_id]

        result_b = model_b_by_frame[frame_id]

        if (
            result_a["gpu_pointer"]
            == result_b["gpu_pointer"]
        ):

            shared_pointer_matches += 1

        if (
            result_a["array_object_id"]
            == result_b["array_object_id"]
        ):

            shared_object_matches += 1

    streams_are_independent = (

        worker_info.get("model_a", {}).get("stream_pointer")
        is not None

        and worker_info.get("model_b", {}).get("stream_pointer")
        is not None

        and worker_info["model_a"]["stream_pointer"]
        != worker_info["model_b"]["stream_pointer"]
    )

    topics_are_independent = (

        topics["model_a"] != topics["model_b"]
    )

    # --------------------------------------------------------
    # KAFKA DELIVERY VALIDATION
    # --------------------------------------------------------

    def publisher_is_valid(model_id):

        stats = publisher_stats[model_id]

        expected_model_id = (
            EXPECTED_RESULT_MODEL_IDS[model_id]
        )

        return all([

            stats["model_id"] == expected_model_id,

            stats["topic"] == topics[model_id],

            stats["enqueued"] == summary["frames_submitted"],

            stats["produce_accepted"] == summary["frames_submitted"],

            stats["delivered"] == summary["frames_submitted"],

            stats["delivery_failures"] == 0,

            stats["publish_failures"] == 0,

            stats["enqueue_timeouts"] == 0,

            stats["worker_errors"] == 0,

            stats["shutdown_pending"] == 0,

            stats["outstanding_after_produce"] == 0,

            stats["queue_depth"] == 0,

        ])

    model_a_kafka_valid = publisher_is_valid(
        "model_a"
    )

    model_b_kafka_valid = publisher_is_valid(
        "model_b"
    )

    # --------------------------------------------------------
    # ARCHITECTURE VALIDATION
    # --------------------------------------------------------

    print("\n" + "=" * 72)

    print("ARCHITECTURE VALIDATION")

    print("=" * 72)

    print(
        f"Matching frame IDs: {matching_frames}"
    )

    print(
        f"Shared GPU pointer matches: "
        f"{shared_pointer_matches}/{summary['frames_submitted']}"
    )

    print(
        f"Shared CuPy object matches: "
        f"{shared_object_matches}/{summary['frames_submitted']}"
    )

    print(
        "All frames fully released: "
        f"{summary['frames_fully_released'] == summary['frames_cloned']}"
    )

    print(
        f"Independent CUDA streams: {streams_are_independent}"
    )

    print(
        f"Independent Kafka topics: {topics_are_independent}"
    )

    print(
        f"Model A Kafka delivery validation: {model_a_kafka_valid}"
    )

    print(
        f"Model B Kafka delivery validation: {model_b_kafka_valid}"
    )

    print(
        "Binary Protobuf consumer verification: "
        f"{kafka_verification['passed']}"
    )

    print(
        f"Verified Protobuf messages: "
        f"{kafka_verification['messages_received']}"
    )

    if kafka_verification.get("error"):
        print(
            f"Protobuf verification error: "
            f"{kafka_verification['error']}"
        )

    print(
        f"Worker errors: {len(worker_errors)}"
    )

    print(
        f"Startup errors: {len(startup_errors)}"
    )

    # --------------------------------------------------------
    # GPU MEMORY
    # --------------------------------------------------------

    free_bytes, total_bytes = (
        cp.cuda.runtime.memGetInfo()
    )

    print("\n========== GPU MEMORY ==========")

    print(
        f"Total: {total_bytes / 1024**2:.2f} MiB"
    )

    print(
        f"Free: {free_bytes / 1024**2:.2f} MiB"
    )

    print(
        f"CuPy pool used: "
        f"{cp.get_default_memory_pool().used_bytes() / 1024**2:.2f} MiB"
    )

    # --------------------------------------------------------
    # FINAL SUCCESS CRITERIA
    # --------------------------------------------------------

    successful = all([

        pipeline_error is None,

        summary["errors"] == 0,

        summary["frames_submitted"] > 0,

        summary["frames_cloned"] == summary["frames_submitted"],

        summary["frames_fully_released"] == summary["frames_cloned"],

        matching_frames,

        shared_pointer_matches == summary["frames_submitted"],

        shared_object_matches == summary["frames_submitted"],

        len(results["model_a"]) == summary["frames_submitted"],

        len(results["model_b"]) == summary["frames_submitted"],

        len(model_a_by_frame) == len(results["model_a"]),

        len(model_b_by_frame) == len(results["model_b"]),

        streams_are_independent,

        topics_are_independent,

        model_a_kafka_valid,

        model_b_kafka_valid,

        kafka_verification["passed"],

        len(worker_errors) == 0,

        len(startup_errors) == 0,

    ])

    print("\n" + "=" * 72)

    if successful:

        print(
            "DEEPSTREAM + TWO GPU MODELS + "
            "INDEPENDENT BINARY PROTOBUF KAFKA TEST PASSED"
        )

    else:

        print(
            "DEEPSTREAM + TWO GPU MODELS + "
            "INDEPENDENT BINARY PROTOBUF KAFKA TEST FAILED"
        )

        for error in worker_errors:

            print(f"WORKER ERROR: {error}")

        for error in startup_errors:

            print(f"STARTUP ERROR: {error}")

        if pipeline_error is not None:

            print(f"PIPELINE ERROR: {pipeline_error}")

    print("=" * 72)

    if not successful:

        raise RuntimeError(
            "DeepStream GPU inference and binary Protobuf Kafka "
            "validation failed."
        )


if __name__ == "__main__":

    main()
