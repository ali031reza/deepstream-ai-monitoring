"""Single-camera, single-mock-model DeepStream/Kafka integration test.

This first milestone intentionally uses one MP4 source and one synthetic GPU
model. It validates frame ownership, GPU inference, Protobuf serialization,
and delivery to exactly one Kafka result topic. RTSP and real model weights
are later milestones.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import cupy as cp
from confluent_kafka import Consumer, KafkaError, KafkaException
from pyservicemaker import BufferRetriever, Flow, Pipeline

from configs.settings import AppSettings, ModelSettings, load_settings
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
MODEL_CLASSES = {
    "model_a": MockGPUModelA,
    "model_b": MockGPUModelB,
}


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value cannot be negative")
    return number


@dataclass(slots=True)
class FramePacket:
    """Own one cloned DeepStream GPU frame until the model releases it."""

    frame_id: int
    captured_at_ms: int
    deepstream_tensor: Any
    gpu_array: Any
    gpu_pointer: int
    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    on_release: Callable[[], None]
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _released: bool = field(default=False, init=False, repr=False)

    def get_gpu_array(self):
        with self._lock:
            if self._released or self.gpu_array is None:
                raise RuntimeError(f"Frame {self.frame_id} has already been released")
            return self.gpu_array

    @property
    def released(self) -> bool:
        with self._lock:
            return self._released

    def release(self) -> None:
        with self._lock:
            if self._released:
                raise RuntimeError(f"Frame {self.frame_id} released more than once")
            self.gpu_array = None
            self.deepstream_tensor = None
            self._released = True
            callback = self.on_release
        callback()


@dataclass(slots=True)
class FrameTask:
    packet: FramePacket
    enqueued_ns: int | None = None
    _enqueued: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def mark_enqueued(self) -> None:
        self.enqueued_ns = time.perf_counter_ns()
        self._enqueued.set()

    def queue_wait_ms(self) -> float:
        if not self._enqueued.wait(timeout=30):
            raise TimeoutError("Frame task was not marked as enqueued")
        assert self.enqueued_ns is not None
        return (time.perf_counter_ns() - self.enqueued_ns) / 1_000_000.0


@dataclass(slots=True)
class RetrieverStats:
    frames_seen: int = 0
    frames_cloned: int = 0
    frames_submitted: int = 0
    frames_released: int = 0
    errors: int = 0
    max_queue_depth: int = 0


class SingleModelRetriever(BufferRetriever):
    """Clone each DeepStream frame once and enqueue it for one model worker."""

    def __init__(self, *, gpu_id: int, frame_queue: queue.Queue,
                 sample_limit: int = 5):
        super().__init__()
        self.gpu_id = gpu_id
        self.frame_queue = frame_queue
        self.sample_limit = sample_limit
        self.stats = RetrieverStats()
        self._lock = threading.Lock()

    def _record_release(self) -> None:
        with self._lock:
            self.stats.frames_released += 1

    def consume(self, buffer) -> int:
        with self._lock:
            frame_id = self.stats.frames_seen
            self.stats.frames_seen += 1

        raw_tensor = None
        owned_tensor = None
        gpu_array = None
        packet = None

        try:
            raw_tensor = buffer.extract(0)
            if raw_tensor is None:
                raise RuntimeError("DeepStream buffer did not contain a frame tensor")

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
                    raise RuntimeError("CuPy frame is on the wrong GPU")
                if gpu_array.dtype != cp.uint8:
                    raise RuntimeError(f"Expected uint8 frame, got {gpu_array.dtype}")
                if gpu_array.ndim != 3 or gpu_array.shape[2] != 3:
                    raise RuntimeError(
                        f"Expected frame shape (H, W, 3), got {gpu_array.shape}"
                    )
                cp.cuda.runtime.deviceSynchronize()

            packet = FramePacket(
                frame_id=frame_id,
                captured_at_ms=time.time_ns() // 1_000_000,
                deepstream_tensor=owned_tensor,
                gpu_array=gpu_array,
                gpu_pointer=int(gpu_array.data.ptr),
                shape=tuple(gpu_array.shape),
                dtype=str(gpu_array.dtype),
                nbytes=int(gpu_array.nbytes),
                on_release=self._record_release,
            )
            owned_tensor = None
            gpu_array = None

            with self._lock:
                self.stats.frames_cloned += 1

            task = FrameTask(packet)
            self.frame_queue.put(task)
            task.mark_enqueued()

            with self._lock:
                self.stats.frames_submitted += 1
                self.stats.max_queue_depth = max(
                    self.stats.max_queue_depth, self.frame_queue.qsize()
                )

            if frame_id < self.sample_limit:
                print(
                    f"[CAMERA] frame={frame_id} shape={packet.shape} "
                    f"gpu={self.gpu_id} bytes={packet.nbytes} "
                    "submitted_to=one_model",
                    flush=True,
                )
            elif (frame_id + 1) % 100 == 0:
                print(f"[CAMERA] received {frame_id + 1} frames", flush=True)
            return 1

        except Exception as exc:
            with self._lock:
                self.stats.errors += 1
            if packet is not None and not packet.released:
                try:
                    packet.release()
                except Exception:
                    traceback.print_exc()
            print(f"[CAMERA ERROR] frame={frame_id}: {exc}", flush=True)
            traceback.print_exc()
            return -1

        finally:
            if raw_tensor is not None:
                del raw_tensor
            if owned_tensor is not None:
                del owned_tensor
            if gpu_array is not None:
                del gpu_array

    def summary(self) -> dict[str, int]:
        with self._lock:
            return {
                "frames_seen": self.stats.frames_seen,
                "frames_cloned": self.stats.frames_cloned,
                "frames_submitted": self.stats.frames_submitted,
                "frames_released": self.stats.frames_released,
                "errors": self.stats.errors,
                "max_queue_depth": self.stats.max_queue_depth,
            }


@dataclass(slots=True)
class SingleModelWorker:
    """Own one mock model instance, one CUDA stream, and one Kafka publisher."""

    model_key: str
    model_settings: ModelSettings
    gpu_id: int
    camera_id: str
    source_session_id: str
    frame_queue: queue.Queue
    publisher: AsyncProtobufKafkaPublisher
    ready_event: threading.Event
    results: list[dict[str, Any]] = field(default_factory=list)
    worker_errors: list[str] = field(default_factory=list)
    startup_errors: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def run(self) -> None:
        stream = None
        model = None
        initialized = False

        try:
            started = time.perf_counter()
            with cp.cuda.Device(self.gpu_id):
                stream = cp.cuda.Stream(non_blocking=True)
                model_config = {
                    "input_size": self.model_settings.input_size,
                    "hidden_units": self.model_settings.hidden_units,
                    "top_k": self.model_settings.top_k,
                    "seed": self.model_settings.seed,
                }
                model = MODEL_CLASSES[self.model_key](
                    cp=cp, gpu_id=self.gpu_id, stream=stream, **model_config
                )
                stream.synchronize()

            initialization_ms = (time.perf_counter() - started) * 1000.0
            self.info.update({
                "gpu_id": self.gpu_id,
                "stream_pointer": int(stream.ptr),
                "initialization_ms": initialization_ms,
                "model_config": model_config,
            })
            initialized = True
            self.ready_event.set()
            print(
                f"[MODEL] {self.model_settings.model_id} ready on GPU "
                f"{self.gpu_id}; CUDA stream={stream.ptr}; "
                f"init={initialization_ms:.2f} ms",
                flush=True,
            )

            while True:
                task = self.frame_queue.get()
                try:
                    if task is None:
                        print("[MODEL] shutdown received", flush=True)
                        return
                    self._process_task(task, model, stream)
                finally:
                    self.frame_queue.task_done()

        except BaseException as exc:
            message = f"{type(exc).__name__}: {exc}"
            if initialized:
                self.worker_errors.append(message)
            else:
                self.startup_errors.append(message)
            traceback.print_exc()
            self._drain_pending()

        finally:
            self.ready_event.set()
            if stream is not None:
                try:
                    stream.synchronize()
                except Exception as exc:
                    self.worker_errors.append(
                        f"Final CUDA synchronization failed: {exc}"
                    )
            model = None
            stream = None

    def _process_task(self, task: FrameTask, model, stream) -> None:
        packet = task.packet
        frame = None
        publish_failed = False

        try:
            queue_wait_ms = task.queue_wait_ms()
            frame = packet.get_gpu_array()
            frame_pointer = int(frame.data.ptr)
            frame_object_id = id(frame)

            started = time.perf_counter_ns()
            output = model.infer(frame, frame_id=packet.frame_id)
            stream.synchronize()
            processing_ms = (time.perf_counter_ns() - started) / 1_000_000.0

            del frame
            frame = None
            packet.release()

            if output.get("model_id") != self.model_settings.model_id:
                raise RuntimeError(
                    f"Model returned {output.get('model_id')!r}; "
                    f"expected {self.model_settings.model_id!r}"
                )
            if int(output.get("frame_id", -1)) != packet.frame_id:
                raise RuntimeError("Model result frame_id does not match camera frame")

            result_data = dict(output)
            result_data["input_frame_shape"] = list(packet.shape)
            timings = dict(output.get("timings_ms", {}))
            timings["queue_wait_ms"] = queue_wait_ms
            timings["worker_processing_ms"] = processing_ms
            result_data["timings_ms"] = timings

            message = build_inference_result(
                result_data,
                camera_id=self.camera_id,
                source_session_id=self.source_session_id,
                frame_sequence=packet.frame_id,
                captured_at_ms=packet.captured_at_ms,
                worker_id=f"{self.model_key}-worker-01",
            )

            try:
                self.publisher.publish(message)
            except Exception:
                publish_failed = True
                raise

            self.results.append({
                "frame_sequence": packet.frame_id,
                "frame_id": message.frame_id,
                "result_id": message.result_id,
                "gpu_pointer": frame_pointer,
                "array_object_id": frame_object_id,
                "detection_count": len(message.detections),
                "timings_ms": timings,
            })

        except Exception as exc:
            self.worker_errors.append(
                f"frame={packet.frame_id}: {type(exc).__name__}: {exc}"
            )
            traceback.print_exc()
            if publish_failed:
                raise RuntimeError(
                    "Kafka publishing failed; stopping the model worker "
                    "to avoid silently losing inference results"
                ) from exc

        finally:
            if frame is not None:
                del frame
            if not packet.released:
                try:
                    packet.release()
                except Exception as exc:
                    self.worker_errors.append(
                        f"frame={packet.frame_id}: frame release failed: {exc}"
                    )

    def _drain_pending(self) -> None:
        while True:
            try:
                task = self.frame_queue.get_nowait()
            except queue.Empty:
                return
            try:
                if task is not None and not task.packet.released:
                    task.packet.release()
            except Exception as exc:
                self.worker_errors.append(f"Queue-drain release failed: {exc}")
            finally:
                self.frame_queue.task_done()


@dataclass(slots=True)
class KafkaResultVerifier:
    consumer: Consumer
    topic: str
    model_id: str
    camera_id: str
    source_session_id: str

    def wait_for_assignment(self, timeout_seconds: float = 30.0) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            assignment = self.consumer.assignment()
            if assignment:
                message = self.consumer.poll(0.1)
                if message is not None and message.error():
                    if message.error().code() != KafkaError._PARTITION_EOF:
                        raise KafkaException(message.error())
                print(
                    f"[KAFKA] verifier assigned {len(assignment)} partition(s)",
                    flush=True,
                )
                return

            message = self.consumer.poll(0.1)
            if message is not None and message.error():
                if message.error().code() != KafkaError._PARTITION_EOF:
                    raise KafkaException(message.error())

        raise TimeoutError("Kafka verifier did not receive a partition assignment")

    def verify(self, expected_frames: int, timeout_seconds: float = 60.0) -> dict:
        received: dict[int, InferenceResult] = {}
        deadline = time.monotonic() + timeout_seconds

        while len(received) < expected_frames:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            message = self.consumer.poll(min(1.0, remaining))
            if message is None:
                continue
            if message.error():
                if message.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(message.error())
            if message.topic() != self.topic or not message.value():
                continue

            decoded = InferenceResult()
            decoded.ParseFromString(message.value())

            if decoded.source_session_id != self.source_session_id:
                continue
            if decoded.model_id != self.model_id:
                raise AssertionError("Kafka result model_id mismatch")
            if decoded.camera_id != self.camera_id:
                raise AssertionError("Kafka result camera_id mismatch")
            if decoded.schema_version != "1.0" or decoded.status != "processed":
                raise AssertionError("Kafka result schema/status mismatch")

            sequence = int(decoded.frame_sequence)
            if not 0 <= sequence < expected_frames:
                raise AssertionError(f"Unexpected frame sequence {sequence}")
            expected_frame_id = f"{self.source_session_id}:{sequence:012d}"
            if decoded.frame_id != expected_frame_id:
                raise AssertionError("Kafka frame_id does not match session/sequence")
            if not decoded.result_id:
                raise AssertionError("Kafka result is missing result_id")
            if decoded.captured_at_ms <= 0 or decoded.processed_at_ms <= 0:
                raise AssertionError("Kafka result is missing timestamps")
            if not decoded.detections:
                raise AssertionError("Mock model result contains no detections")

            timing_names = {item.stage for item in decoded.timings}
            if not {"queue_wait_ms", "worker_processing_ms"}.issubset(timing_names):
                raise AssertionError("Kafka result is missing worker timing metadata")
            if sequence in received:
                raise AssertionError(f"Duplicate Kafka result for frame {sequence}")

            received[sequence] = decoded

        expected = set(range(expected_frames))
        actual = set(received)
        if actual != expected:
            missing = sorted(expected - actual)
            raise AssertionError(
                f"Kafka received {len(actual)}/{expected_frames} results; "
                f"missing={missing[:10]}"
            )

        result_ids = [item.result_id for item in received.values()]
        if len(result_ids) != len(set(result_ids)):
            raise AssertionError("Kafka result IDs are not unique")

        return {
            "passed": True,
            "messages_received": len(received),
            "topic": self.topic,
            "model_id": self.model_id,
            "camera_id": self.camera_id,
        }


@dataclass(frozen=True, slots=True)
class SingleModelOptions:
    source: Path
    camera_id: str
    gpu_id: int
    queue_size: int
    sample_limit: int
    kafka_config: Path
    result_queue_size: int | None = None
    batch_size: int | None = None
    flush_interval_ms: int | None = None


@dataclass(slots=True)
class SingleModelCameraApplication:
    """Own the one-camera, one-model mock inference test lifecycle."""

    settings: AppSettings
    options: SingleModelOptions

    def _active_model(self) -> tuple[str, ModelSettings]:
        active = [
            (key, value) for key, value in self.settings.models.items()
            if value.enabled
        ]
        if len(active) != 1:
            raise ValueError(
                "This branch requires exactly one enabled model in configs/config.yaml"
            )
        return active[0]

    def run(self) -> dict:
        options = self.options
        if not options.source.is_file():
            raise FileNotFoundError(f"Video file not found: {options.source}")
        if not options.camera_id.strip():
            raise ValueError("Camera ID must not be empty")
        if options.queue_size < 1:
            raise ValueError("Input queue size must be positive")

        model_key, model_settings = self._active_model()
        if model_settings.workers != 1:
            raise ValueError(
                "The initial one-model mock test uses exactly one model worker; "
                "set that model's workers value to 1"
            )

        gpu_count = cp.cuda.runtime.getDeviceCount()
        if options.gpu_id >= gpu_count:
            raise RuntimeError(
                f"GPU {options.gpu_id} unavailable; detected {gpu_count} GPU(s)"
            )

        with cp.cuda.Device(options.gpu_id):
            properties = cp.cuda.runtime.getDeviceProperties(options.gpu_id)
        gpu_name = properties["name"]
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()

        kafka_config = load_kafka_config(options.kafka_config)
        topic = topic_for_model(kafka_config, model_settings.model_id)
        publisher_config = producer_config_for_model(
            kafka_config, model_settings.model_id
        )
        publisher_kwargs = publisher_options(kafka_config)
        if options.result_queue_size is not None:
            publisher_kwargs["queue_maxsize"] = options.result_queue_size
        if options.batch_size is not None:
            publisher_kwargs["max_batch_messages"] = options.batch_size
        if options.flush_interval_ms is not None:
            publisher_kwargs["batch_flush_interval_ms"] = options.flush_interval_ms

        session_id = uuid4().hex
        consumer = Consumer({
            "bootstrap.servers": kafka_config["bootstrap_servers"],
            "group.id": f"single-model-camera-verify-{session_id}",
            "auto.offset.reset": "latest",
            "max.poll.interval.ms": 86_400_000,
            "enable.auto.commit": False,
        })
        consumer.subscribe([topic])
        verifier = KafkaResultVerifier(
            consumer=consumer,
            topic=topic,
            model_id=model_settings.model_id,
            camera_id=options.camera_id,
            source_session_id=session_id,
        )

        publisher = AsyncProtobufKafkaPublisher(
            model_id=model_settings.model_id,
            topic=topic,
            producer_config=publisher_config,
            **publisher_kwargs,
        )
        frame_queue: queue.Queue = queue.Queue(maxsize=options.queue_size)
        retriever = SingleModelRetriever(
            gpu_id=options.gpu_id,
            frame_queue=frame_queue,
            sample_limit=options.sample_limit,
        )
        worker = SingleModelWorker(
            model_key=model_key,
            model_settings=model_settings,
            gpu_id=options.gpu_id,
            camera_id=options.camera_id,
            source_session_id=session_id,
            frame_queue=frame_queue,
            publisher=publisher,
            ready_event=threading.Event(),
        )
        worker_thread = threading.Thread(
            target=worker.run,
            name=f"{model_key.replace('_', '-')}-worker",
            daemon=False,
        )

        publisher_started = False
        worker_started = False
        publisher_stats: dict[str, Any] = {}
        kafka_verification: dict[str, Any] = {"passed": False, "messages_received": 0}
        pipeline_error: BaseException | None = None
        publisher_error: str | None = None
        pipeline_started = 0.0

        print("=" * 72)
        print("DEEPSTREAM + ONE CAMERA + ONE MOCK GPU MODEL + PROTOBUF KAFKA")
        print("=" * 72)
        print(f"Video: {options.source}")
        print(f"Camera ID: {options.camera_id}")
        print(f"Source session ID: {session_id}")
        print(f"GPU: {gpu_name} (ID {options.gpu_id})")
        print(f"Model: {model_settings.model_id}")
        print(f"Kafka topic: {topic}")
        print(f"Input queue capacity: {options.queue_size}")
        print("Model workers: 1")
        print("Output: binary Protobuf")

        try:
            verifier.wait_for_assignment()
            publisher.start()
            publisher_started = True
            worker_thread.start()
            worker_started = True

            if not worker.ready_event.wait(timeout=60):
                raise TimeoutError("Mock GPU model did not initialize within 60 seconds")
            if worker.startup_errors:
                raise RuntimeError(f"Mock model initialization failed: {worker.startup_errors}")

            print("[READY] Camera retriever, mock model, and Kafka publisher are ready",
                  flush=True)
            flow = (
                Flow(Pipeline("single-model-one-camera"))
                .capture([str(options.source)], gpu_id=options.gpu_id)
                .retrieve(retriever)
            )
            pipeline_started = time.perf_counter()
            flow()

        except BaseException as exc:
            pipeline_error = exc
            print(f"[PIPELINE ERROR] {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()

        finally:
            if worker_started and worker_thread.is_alive():
                frame_queue.put(None)
            if worker_started:
                frame_queue.join()
                worker_thread.join(timeout=60)
                if worker_thread.is_alive():
                    worker.worker_errors.append(
                        "Model worker did not stop within 60 seconds"
                    )

            try:
                publisher_stats = publisher.close()
            except Exception as exc:
                publisher_error = f"{type(exc).__name__}: {exc}"
                publisher_stats = publisher.stats()
                print(f"[KAFKA SHUTDOWN ERROR] {publisher_error}", flush=True)

        elapsed = (
            time.perf_counter() - pipeline_started
            if pipeline_started else 0.0
        )
        summary = retriever.summary()
        if summary["frames_submitted"] > 0 and publisher_error is None:
            try:
                kafka_verification = verifier.verify(
                    summary["frames_submitted"], timeout_seconds=60
                )
            except Exception as exc:
                kafka_verification["error"] = f"{type(exc).__name__}: {exc}"
                traceback.print_exc()
        else:
            kafka_verification["error"] = "No submitted frames to verify"

        consumer.close()

        expected_frames = summary["frames_submitted"]
        result_sequences = [item["frame_sequence"] for item in worker.results]
        unique_sequences = set(result_sequences)
        success = all([
            pipeline_error is None,
            publisher_error is None,
            summary["errors"] == 0,
            expected_frames > 0,
            summary["frames_cloned"] == expected_frames,
            summary["frames_released"] == summary["frames_cloned"],
            len(worker.results) == expected_frames,
            unique_sequences == set(range(expected_frames)),
            len(unique_sequences) == len(result_sequences),
            publisher_stats.get("enqueued") == expected_frames,
            publisher_stats.get("produce_accepted") == expected_frames,
            publisher_stats.get("delivered") == expected_frames,
            publisher_stats.get("delivery_failures") == 0,
            publisher_stats.get("publish_failures") == 0,
            publisher_stats.get("enqueue_timeouts") == 0,
            publisher_stats.get("worker_errors") == 0,
            publisher_stats.get("shutdown_pending") == 0,
            publisher_stats.get("outstanding_after_produce") == 0,
            publisher_stats.get("queue_depth") == 0,
            kafka_verification.get("passed") is True,
            len(worker.worker_errors) == 0,
            len(worker.startup_errors) == 0,
        ])

        print("\n" + "=" * 72)
        print("SINGLE-MODEL CAMERA TEST SUMMARY")
        print("=" * 72)
        print(f"Frames received: {summary['frames_seen']}")
        print(f"GPU frame clones: {summary['frames_cloned']}")
        print(f"Frames submitted: {summary['frames_submitted']}")
        print(f"Frames released: {summary['frames_released']}")
        print(f"Model results: {len(worker.results)}")
        print(f"Kafka topic: {topic}")
        print(f"Kafka delivered: {publisher_stats.get('delivered', 0)}")
        print(f"Kafka verified: {kafka_verification.get('messages_received', 0)}")
        print(f"Delivery failures: {publisher_stats.get('delivery_failures', 0)}")
        print(f"Worker errors: {len(worker.worker_errors)}")
        print(f"Pipeline elapsed: {elapsed:.3f} seconds")
        if elapsed > 0:
            print(f"Submitted throughput: {expected_frames / elapsed:.2f} FPS")
        if worker.info:
            print(f"CUDA stream: {worker.info.get('stream_pointer')}")
            print(f"Model initialization: {worker.info.get('initialization_ms', 0):.2f} ms")
        if worker.results:
            print(f"Sample detection count: {worker.results[0]['detection_count']}")
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        print(f"GPU memory free: {free_bytes / 1024**2:.2f} / {total_bytes / 1024**2:.2f} MiB")
        if pipeline_error is not None:
            print(f"Pipeline error: {pipeline_error}")
        if publisher_error:
            print(f"Publisher error: {publisher_error}")
        for error in worker.startup_errors + worker.worker_errors:
            print(f"WORKER ERROR: {error}")

        print("=" * 72)
        if success:
            print("DEEPSTREAM + ONE CAMERA + ONE MOCK MODEL + KAFKA TEST PASSED")
        else:
            print("DEEPSTREAM + ONE CAMERA + ONE MOCK MODEL + KAFKA TEST FAILED")
        print("=" * 72)

        if not success:
            raise RuntimeError("Single-model camera integration test failed")

        return {
            "source_session_id": session_id,
            "camera_id": options.camera_id,
            "model_id": model_settings.model_id,
            "topic": topic,
            "frames_processed": expected_frames,
            "kafka_messages_verified": kafka_verification["messages_received"],
            "elapsed_seconds": elapsed,
        }


def build_parser(settings: AppSettings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=settings.source.resolved_uri(settings.paths))
    parser.add_argument("--source-id", default=settings.source.camera_id)
    parser.add_argument("--gpu-id", type=nonnegative_int, default=settings.gpu_id)
    parser.add_argument("--queue-size", type=positive_int,
                        default=settings.queues.model_input_capacity)
    parser.add_argument("--sample-limit", type=nonnegative_int,
                        default=settings.sample_limit)
    parser.add_argument("--kafka-config", default=str(settings.paths.config_file))
    parser.add_argument("--result-queue-size", type=positive_int, default=None)
    parser.add_argument("--batch-size", type=positive_int, default=None)
    parser.add_argument("--flush-interval-ms", type=positive_int, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    settings = load_settings()
    active_models = [model for model in settings.models.values() if model.enabled]
    if len(active_models) != 1:
        raise SystemExit(
            "Enable exactly one model in configs/config.yaml before running this branch"
        )
    if settings.source.mode != "file":
        raise SystemExit(
            "This initial mock test accepts an MP4 file; RTSP camera ingestion is a later step"
        )

    parser = build_parser(settings)
    args = parser.parse_args(argv)
    options = SingleModelOptions(
        source=settings.paths.resolve(args.source),
        camera_id=args.source_id,
        gpu_id=args.gpu_id,
        queue_size=args.queue_size,
        sample_limit=args.sample_limit,
        kafka_config=settings.paths.resolve(args.kafka_config),
        result_queue_size=args.result_queue_size,
        batch_size=args.batch_size,
        flush_interval_ms=args.flush_interval_ms,
    )
    SingleModelCameraApplication(settings, options).run()


if __name__ == "__main__":
    main()
