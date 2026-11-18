"""One-camera DeepStream + YOLO11n TensorRT + event-frame/Kafka pipeline.

Run inside the DeepStream 9.2 container. Frames are sampled by source PTS,
then sent to a TensorRT FP16 engine. Qualifying person frames are atomically
uploaded as paired JPEG objects before their Protobuf result is published.
"""
from __future__ import annotations

from collections import Counter
import argparse
from dataclasses import asdict, dataclass, field
import json
import os
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import yaml
from confluent_kafka import Consumer, KafkaError, KafkaException, TopicPartition

from configs.settings import AppSettings, load_settings
from src.kafka.message_validator import validate_inference_result
from src.kafka.topic_manager import KafkaTopicManager
from src.models.tensorrt_yolo11n_engine import TensorRTYOLO11nEngine
from src.pipeline.throughput import rate
from src.pipeline.frame_sampler import FrameSampler
from src.proto.inference_result_pb2 import InferenceResult
from src.results.async_protobuf_kafka_publisher import AsyncProtobufKafkaPublisher
from src.results.kafka_config import (
    load_kafka_config,
    producer_config_for_model,
    publisher_options,
    topic_for_model,
)
from src.results.protobuf_result_adapter import build_inference_result
from src.storage.event_frame_store import EventFrameStore


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be positive")
    return number


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value cannot be negative")
    return number


@dataclass(frozen=True, slots=True)
class TensorRTEventPipelineOptions:
    source: Path
    camera_id: str
    gpu_id: int
    sample_fps: float
    fallback_source_fps: float
    queue_size: int
    batch_width: int
    batch_height: int
    engine_path: Path
    event_root: str
    event_min_confidence: float
    detection_class: str
    event_every_nth_sample: int
    event_type: str
    result_topic: str
    kafka_bootstrap_servers: str
    result_root: Path
    jpeg_quality: int
    session_id: str
    max_samples: int | None = None
    verification_timeout: float = 90.0

    def __post_init__(self) -> None:
        if not self.verification_timeout > 0:
            raise ValueError("verification timeout must be positive")
        if not self.camera_id.strip() or len(self.camera_id) > 50:
            raise ValueError("camera_id must contain 1-50 characters")
        if self.gpu_id < 0 or self.queue_size < 1:
            raise ValueError("gpu_id must be nonnegative and queue_size positive")
        if self.batch_width < 1 or self.batch_height < 1:
            raise ValueError("DeepStream batch width and height must be positive")
        if self.sample_fps <= 0 or self.fallback_source_fps <= 0:
            raise ValueError("sample_fps and fallback_source_fps must be positive")
        if not 0 <= self.event_min_confidence <= 1:
            raise ValueError("event_min_confidence must be in [0, 1]")
        if not self.detection_class.strip():
            raise ValueError("detection_class is required")
        if self.event_every_nth_sample < 1:
            raise ValueError("event_every_nth_sample must be >= 1")
        if not self.event_type.strip() or not self.result_topic.strip():
            raise ValueError("event_type and result_topic are required")
        if not self.kafka_bootstrap_servers.strip():
            raise ValueError("Kafka bootstrap servers are required")
        if not 1 <= self.jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be between 1 and 95")
        if self.max_samples is not None and self.max_samples < 1:
            raise ValueError("max_samples must be positive when supplied")


@dataclass(slots=True)
class FramePacket:
    """Own a cloned DeepStream GPU frame until inference copies it to host."""

    frame_sequence: int
    sample_index: int
    source_timestamp_ms: float
    captured_at_ms: int
    deepstream_tensor: Any
    gpu_array: Any
    shape: tuple[int, ...]
    on_release: Any
    _released: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    @property
    def released(self) -> bool:
        with self._lock:
            return self._released

    def release(self) -> None:
        with self._lock:
            if self._released:
                raise RuntimeError(
                    f"Frame {self.frame_sequence} was released more than once"
                )
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
    frames_sampled: int = 0
    frames_skipped: int = 0
    frames_cloned: int = 0
    frames_released: int = 0
    errors: int = 0
    max_queue_depth: int = 0


@dataclass(slots=True)
class DeepStreamFrameRetrieverCore:
    """Sample by PTS before cloning and enqueue only selected frames."""

    cp: Any
    gpu_id: int
    frame_queue: queue.Queue
    sampler: FrameSampler
    fatal_event: threading.Event
    max_samples: int | None = None
    sample_log_limit: int = 5
    stats: RetrieverStats = field(default_factory=RetrieverStats)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def _record_release(self) -> None:
        with self._lock:
            self.stats.frames_released += 1

    def _enqueue_with_abort(self, task: FrameTask) -> None:
        while True:
            if self.fatal_event.is_set():
                raise RuntimeError("Inference worker failed while waiting to enqueue frame")
            try:
                self.frame_queue.put(task, timeout=0.25)
                task.mark_enqueued()
                return
            except queue.Full:
                continue

    def consume(self, buffer: Any) -> int:
        if self.fatal_event.is_set():
            return -1

        try:
            # Keep the native BatchMetadata wrapper alive
            # throughout the entire metadata iteration.
            batch_meta = buffer.batch_meta

            if batch_meta is None:
                raise RuntimeError(
                    "DeepStream buffer has no batch metadata"
                )

            frame_count = 0

            # IMPORTANT:
            # Iterate directly over the native iterator.
            # Do NOT use list(batch_meta.frame_items).
            for frame_meta in batch_meta.frame_items:

                frame_count += 1

                if self.fatal_event.is_set():
                    return -1

                # Access native metadata while its iterator is alive.
                frame_number = int(frame_meta.frame_number)
                batch_id = int(frame_meta.batch_id)

                raw_pts = getattr(
                    frame_meta,
                    "buffer_pts",
                    None,
                )

                try:
                    pts_ns = (
                        int(raw_pts)
                        if raw_pts is not None
                        else None
                    )
                except (TypeError, ValueError, OverflowError):
                    pts_ns = None

                with self._lock:
                    self.stats.frames_seen += 1
                    sample_count = self.stats.frames_sampled

                if (
                    self.max_samples is not None
                    and sample_count >= self.max_samples
                ):
                    with self._lock:
                        self.stats.frames_skipped += 1
                    continue

                decision = self.sampler.decide(
                    frame_number=frame_number,
                    pts_ns=pts_ns,
                )

                if not decision.should_process:
                    with self._lock:
                        self.stats.frames_skipped += 1
                    continue

                raw_tensor = None
                owned_tensor = None
                gpu_array = None
                packet = None

                try:
                    # Extract the GPU frame from DeepStream.
                    raw_tensor = buffer.extract(batch_id)

                    if raw_tensor is None:
                        raise RuntimeError(
                            f"DeepStream could not extract frame "
                            f"{frame_number}"
                        )

                    # Clone the tensor to establish independent ownership.
                    owned_tensor = raw_tensor.clone()

                    del raw_tensor
                    raw_tensor = None

                    # Convert the owned tensor into a CuPy GPU array.
                    with self.cp.cuda.Device(self.gpu_id):

                        device_type, device_id = tuple(
                            owned_tensor.__dlpack_device__()
                        )

                        if (device_type, device_id) != (
                            2,
                            self.gpu_id,
                        ):
                            raise RuntimeError(
                                f"Unexpected DLPack device "
                                f"{(device_type, device_id)}; "
                                f"expected CUDA device "
                                f"{(2, self.gpu_id)}"
                            )

                        gpu_array = self.cp.from_dlpack(
                            owned_tensor
                        )

                        if gpu_array.dtype != self.cp.uint8:
                            raise RuntimeError(
                                f"Expected uint8 frame, "
                                f"received {gpu_array.dtype}"
                            )

                        if (
                            gpu_array.ndim != 3
                            or gpu_array.shape[2] != 3
                        ):
                            raise RuntimeError(
                                f"Expected RGB HxWx3 frame, "
                                f"got {gpu_array.shape}"
                            )

                        self.cp.cuda.runtime.deviceSynchronize()

                    with self._lock:
                        sample_index = self.stats.frames_sampled

                    # Build the frame packet.
                    packet = FramePacket(
                        frame_sequence=frame_number,
                        sample_index=sample_index,
                        source_timestamp_ms=decision.source_timestamp_ms,
                        captured_at_ms=(
                            time.time_ns() // 1_000_000
                        ),
                        deepstream_tensor=owned_tensor,
                        gpu_array=gpu_array,
                        shape=tuple(gpu_array.shape),
                        on_release=self._record_release,
                    )

                    # Ownership is now transferred to FramePacket.
                    owned_tensor = None
                    gpu_array = None

                    # Enqueue the frame for TensorRT processing.
                    task = FrameTask(packet)

                    self._enqueue_with_abort(task)

                    # Ownership is now transferred to the queue.
                    packet = None

                    with self._lock:
                        self.stats.frames_sampled += 1
                        self.stats.frames_cloned += 1

                        self.stats.max_queue_depth = max(
                            self.stats.max_queue_depth,
                            self.frame_queue.qsize(),
                        )

                        logged_count = self.stats.frames_sampled

                    # Logging.
                    if logged_count <= self.sample_log_limit:

                        print(
                            f"[CAMERA] source_frame={frame_number} "
                            f"sample={sample_index} "
                            f"pts_ms={decision.source_timestamp_ms:.2f} "
                            f"shape={task.packet.shape} "
                            f"gpu={self.gpu_id}",
                            flush=True,
                        )

                    elif logged_count % 25 == 0:

                        print(
                            f"[CAMERA] sampled={logged_count} "
                            f"decoded={self.stats.frames_seen}",
                            flush=True,
                        )

                finally:
                    # Release temporary resources safely.
                    if raw_tensor is not None:
                        del raw_tensor

                    if owned_tensor is not None:
                        del owned_tensor

                    if gpu_array is not None:
                        del gpu_array

                    if (
                        packet is not None
                        and not packet.released
                    ):
                        packet.release()

            # Validate that the batch contained frame metadata.
            if frame_count == 0:
                raise RuntimeError(
                    "DeepStream buffer contains no frame metadata"
                )

            return 1

        except Exception as exc:

            with self._lock:
                self.stats.errors += 1

            self.fatal_event.set()

            print(
                f"[CAMERA ERROR] {type(exc).__name__}: {exc}",
                flush=True,
            )

            traceback.print_exc()

            return -1

    def summary(self) -> dict[str, int]:
        with self._lock:
            return asdict(self.stats)


@dataclass(slots=True)
class JsonlManifestWriter:
    path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict[str, Any]) -> None:
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(payload + "\n")
                handle.flush()


@dataclass(slots=True)
class TensorRTEventWorker:
    settings: AppSettings
    options: TensorRTEventPipelineOptions
    frame_queue: queue.Queue
    publisher: AsyncProtobufKafkaPublisher
    frame_store: EventFrameStore
    manifest_writer: JsonlManifestWriter
    ready_event: threading.Event
    fatal_event: threading.Event
    results: list[dict[str, Any]] = field(default_factory=list)
    startup_errors: list[str] = field(default_factory=list)
    worker_errors: list[str] = field(default_factory=list)
    detector: TensorRTYOLO11nEngine | None = field(default=None, init=False)
    active_ms: Counter = field(default_factory=Counter)
    qualifying_frames: int = 0
    model_load_ms: float = field(default=0.0, init=False)

    def run(self) -> None:
        initialized = False
        try:
            yolo = self.settings.yolo
            class_names = {
                int(class_id): ("person" if int(class_id) == 0 else f"class_{class_id}")
                for class_id in yolo.classes
            }
            self.detector = TensorRTYOLO11nEngine(
                engine_path=str(self.options.engine_path),
                gpu_id=self.options.gpu_id,
                image_size=yolo.image_size,
                confidence=yolo.confidence,
                iou=yolo.iou,
                classes=yolo.classes,
                max_detections=yolo.max_detections,
                class_names=class_names,
            )
            self.model_load_ms = self.detector.load()
            initialized = True
            self.ready_event.set()
            print(
                f"[MODEL] {yolo.model_id} ready on GPU {self.options.gpu_id}; "
                f"TensorRT engine={self.options.engine_path.name}; "
                f"init={self.model_load_ms:.2f} ms",
                flush=True,
            )

            while True:
                task = self.frame_queue.get()
                try:
                    if task is None:
                        return
                    if self.fatal_event.is_set():
                        if not task.packet.released:
                            task.packet.release()
                        continue
                    self._process_task(task)
                finally:
                    self.frame_queue.task_done()

        except BaseException as exc:
            message = f"{type(exc).__name__}: {exc}"
            if initialized:
                self.worker_errors.append(message)
            else:
                self.startup_errors.append(message)
            self.fatal_event.set()
            traceback.print_exc()
            self._drain_pending()

        finally:
            self.ready_event.set()
            self._drain_pending()
            if self.detector is not None:
                try:
                    self.detector.close()
                except Exception as exc:
                    self.worker_errors.append(
                        f"TensorRT shutdown failed: {type(exc).__name__}: {exc}"
                    )

    def _process_task(self, task: FrameTask) -> None:
        packet = task.packet
        try:
            queue_wait_ms = task.queue_wait_ms()
            assert self.detector is not None
            inference = self.detector.infer(packet.gpu_array)
            # infer() copied the RGB frame to host and synchronized TensorRT.
            packet.release()

            detections = [item.to_payload() for item in inference.detections]
            qualifying = [
                item for item in inference.detections
                if item.class_id in self.settings.yolo.classes
                and item.class_name.casefold() == self.options.detection_class.casefold()
                and item.confidence >= self.options.event_min_confidence
            ]
            is_event = (
                self.settings.events.enabled
                and bool(qualifying)
                and packet.sample_index % self.options.event_every_nth_sample == 0
            )

            processed_at_ms = time.time_ns() // 1_000_000
            image = None
            boxed_metadata = None
            event_storage_ms = 0.0

            if is_event:
                store_started = time.perf_counter()
                frame_args = dict(
                    source_session_id=self.options.session_id,
                    frame_sequence=packet.frame_sequence,
                    rgb_array=inference.frame_rgb,
                )
                image = self.frame_store.put_rgb(**frame_args)
                boxed = self.frame_store.put_rgb(
                    **frame_args, boxes=[item.bbox_xyxy for item in qualifying]
                )
                boxed_metadata = {
                    "storage_key": boxed.storage_key,
                    "format": boxed.image_format,
                    "size_bytes": boxed.image_size_bytes,
                    "sha256": boxed.image_sha256,
                }
                event_storage_ms = (time.perf_counter() - store_started) * 1000.0

            result_data: dict[str, Any] = {
                "model_id": self.settings.yolo.model_id,
                "frame_id": packet.frame_sequence,
                "gpu_id": self.options.gpu_id,
                "input_frame_shape": list(packet.shape),
                "model_input_size": self.settings.yolo.image_size,
                "status": "processed",
                "is_event": is_event,
                "detections": detections,
                "timings_ms": {
                    "queue_wait_ms": queue_wait_ms,
                    "preprocess_ms": inference.preprocess_ms,
                    "host_to_device_ms": inference.host_to_device_ms,
                    "tensorrt_inference_ms": inference.inference_ms,
                    "postprocess_ms": inference.postprocess_ms,
                    "event_image_store_ms": event_storage_ms,
                },
            }

            if image is not None:
                result_data.update({
                    "event_type": self.options.event_type,
                    "storage_key": image.storage_key,
                    "event_storage_root": self.frame_store.root,
                    "boxed_key": boxed.storage_key,
                    "boxed_sha256": boxed.image_sha256,
                    "boxed_bytes": boxed.image_size_bytes,
                    "source_video_ms": packet.source_timestamp_ms,
                    "image_format": image.image_format,
                    "image_size_bytes": image.image_size_bytes,
                    "image_sha256": image.image_sha256,
                })

            message = build_inference_result(
                result_data,
                camera_id=self.options.camera_id,
                source_session_id=self.options.session_id,
                frame_sequence=packet.frame_sequence,
                captured_at_ms=packet.captured_at_ms,
                worker_id="tensorrt-yolo11n-worker-01",
                processed_at_ms=processed_at_ms,
            )

            manifest_record = {
                "schema_version": "1.0",
                "camera_id": self.options.camera_id,
                "source_session_id": self.options.session_id,
                "frame_id": message.frame_id,
                "frame_sequence": packet.frame_sequence,
                "sample_index": packet.sample_index,
                "source_video_timestamp_ms": round(
                    packet.source_timestamp_ms, 3
                ),
                "captured_at_ms": packet.captured_at_ms,
                "processed_at_ms": processed_at_ms,
                "model_id": self.settings.yolo.model_id,
                "is_event": is_event,
                "event_type": self.options.event_type if is_event else None,
                "detection_count": len(detections),
                "detections": detections,
                "boxed_image": boxed_metadata,
                "event_image": (
                    {
                        "storage_key": image.storage_key,
                        "sha256": image.image_sha256,
                        "size_bytes": image.image_size_bytes,
                    }
                    if image is not None else None
                ),
                "result_id": message.result_id,
            }
            manifest_record["qualifying_detections"] = len(qualifying)
            manifest_start = time.perf_counter()
            self.manifest_writer.write(manifest_record)
            self.active_ms["result_manifest"] += (time.perf_counter() - manifest_start) * 1000

            # Both JPEG objects are durable before Kafka receives
            # the Protobuf result that references them.
            self.publisher.publish(message)

            self.qualifying_frames += int(bool(qualifying))
            self.active_ms.update({
                "preprocess": inference.preprocess_ms,
                "host_to_device": inference.host_to_device_ms,
                "tensorrt": inference.inference_ms,
                "postprocess": inference.postprocess_ms,
                "event_storage": event_storage_ms,
            })
            self.results.append({
                "frame_sequence": packet.frame_sequence,
                "sample_index": packet.sample_index,
                "result_id": message.result_id,
                "is_event": is_event,
                "detection_count": len(detections),
                "event_storage_key": image.storage_key if image else None,
            })

        except Exception as exc:
            self.worker_errors.append(
                f"frame={packet.frame_sequence}: {type(exc).__name__}: {exc}"
            )
            self.fatal_event.set()
            traceback.print_exc()
        finally:
            if not packet.released:
                try:
                    packet.release()
                except Exception as exc:
                    self.worker_errors.append(
                        f"frame={packet.frame_sequence} release failed: {exc}"
                    )
                    self.fatal_event.set()

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
                self.worker_errors.append(
                    f"Pending frame release failed: {type(exc).__name__}: {exc}"
                )
            finally:
                self.frame_queue.task_done()


@dataclass(slots=True)
class KafkaEventVerifier:
    consumer: Consumer
    topic: str
    model_id: str
    camera_id: str
    session_id: str
    frame_store: EventFrameStore

    def wait_for_assignment(self, timeout_seconds: float = 30.0) -> None:
        self.consumer.subscribe([self.topic])
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            assignment = self.consumer.assignment()
            if assignment:
                starts = []
                for partition in assignment:
                    topic_partition = TopicPartition(
                        partition.topic, partition.partition
                    )
                    _low, high = self.consumer.get_watermark_offsets(
                        topic_partition, timeout=10, cached=False
                    )
                    starts.append(TopicPartition(
                        partition.topic, partition.partition, high
                    ))
                # Ignore older messages; this consumer verifies only this run.
                self.consumer.assign(starts)
                print(
                    f"[KAFKA] verifier assigned {len(starts)} partition(s)",
                    flush=True,
                )
                return
            message = self.consumer.poll(0.1)
            if message is not None and message.error():
                if message.error().code() != KafkaError._PARTITION_EOF:
                    raise KafkaException(message.error())
        raise TimeoutError("Kafka event verifier did not receive assignment")

    def verify(
        self,
        expected: list[dict[str, Any]],
        timeout_seconds: float = 90.0,
    ) -> dict[str, Any]:
        expected_by_sequence = {
            int(item["frame_sequence"]): item for item in expected
        }
        received: dict[int, InferenceResult] = {}
        offsets = {}
        deadline = time.monotonic() + timeout_seconds

        while len(received) < len(expected_by_sequence):
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

            result = InferenceResult()
            result.ParseFromString(message.value())
            if result.source_session_id != self.session_id:
                continue
            validate_inference_result(
                result,
                allowed_model_ids={self.model_id},
                require_schema_version=True,
            )
            if result.camera_id != self.camera_id:
                raise AssertionError("Kafka result camera_id mismatch")
            sequence = int(result.frame_sequence)
            if sequence not in expected_by_sequence:
                raise AssertionError(f"Unexpected sampled frame sequence: {sequence}")
            if result.frame_id != f"{self.session_id}:{sequence:012d}":
                raise AssertionError("Kafka frame_id does not match session/sequence")
            if sequence in received:
                raise AssertionError(f"Duplicate Kafka result for frame {sequence}")

            expected_item = expected_by_sequence[sequence]
            if bool(result.is_event) != bool(expected_item["is_event"]):
                raise AssertionError(f"Event flag mismatch for frame {sequence}")
            if result.is_event:
                self.frame_store.verify(
                    storage_key=result.storage_key,
                    image_size_bytes=result.image_size_bytes,
                    image_sha256=result.image_sha256,
                )
                if result.event_storage_root != self.frame_store.root:
                    raise AssertionError("Unexpected camera storage root")
                self.frame_store.verify(storage_key=result.boxed_key,
                                        image_size_bytes=result.boxed_bytes,
                                        image_sha256=result.boxed_sha256)
            partition = message.partition()
            offsets[partition] = max(offsets.get(partition, 0), message.offset() + 1)
            received[sequence] = result

        expected_sequences = set(expected_by_sequence)
        actual_sequences = set(received)
        if actual_sequences != expected_sequences:
            missing = sorted(expected_sequences - actual_sequences)
            raise AssertionError(
                f"Kafka verified {len(actual_sequences)}/{len(expected_sequences)} "
                f"messages; missing frame sequences={missing[:10]}"
            )

        return {
            "passed": True,
            "messages_received": len(received),
            "end_offsets": offsets,
            "event_messages": sum(1 for item in received.values() if item.is_event),
            "topic": self.topic,
            "model_id": self.model_id,
            "camera_id": self.camera_id,
        }


@dataclass(slots=True)
class TensorRTEventPipelineApplication:
    settings: AppSettings
    options: TensorRTEventPipelineOptions

    def run(self) -> dict[str, Any]:
        from pyservicemaker import BufferRetriever, Flow, Pipeline, Probe
        from src.pipeline.frame_metadata_probe import FrameMetadataProbe
        import cupy as cp

        if not self.options.source.is_file():
            raise FileNotFoundError(f"Source video not found: {self.options.source}")
        if not self.options.engine_path.is_file():
            raise FileNotFoundError(
                f"TensorRT engine not found: {self.options.engine_path}"
            )
        if self.settings.yolo.quantize != 16:
            raise ValueError(
                "This runner expects the configured YOLO engine precision to be FP16 "
                "(yolo.quantize: 16)."
            )

        yolo = self.settings.yolo
        output_directory = self.options.result_root / self.options.session_id
        output_directory.mkdir(parents=True, exist_ok=True)
        manifest_path = output_directory / "results.jsonl"
        report_path = output_directory / "performance.json"

        frame_store = EventFrameStore(
            self.options.event_root,
            quality=self.options.jpeg_quality,
        )
        manifest_writer = JsonlManifestWriter(manifest_path)

        config_path = self.settings.paths.resolve(
            os.environ.get("APP_CONFIG_PATH") or self.settings.paths.config_file
        )
        kafka_config = load_kafka_config(config_path)
        kafka_config["bootstrap_servers"] = self.options.kafka_bootstrap_servers
        kafka_config.setdefault("topics", {})[yolo.model_id] = self.options.result_topic
        topic = topic_for_model(kafka_config, yolo.model_id)

        topic_manager = KafkaTopicManager(
            self.options.kafka_bootstrap_servers,
            [topic],
            kafka_config.get("kafka_admin", {}),
        )
        topic_status = topic_manager.ensure_topics()
        print(f"[KAFKA] topic ready: {topic_status}", flush=True)

        consumer = Consumer({
            "bootstrap.servers": self.options.kafka_bootstrap_servers,
            "group.id": f"tensorrt-event-verifier-{self.options.session_id}",
            "auto.offset.reset": "latest",
            "enable.auto.commit": False,
            "max.poll.interval.ms": 86_400_000,
        })
        verifier = KafkaEventVerifier(
            consumer=consumer,
            topic=topic,
            model_id=yolo.model_id,
            camera_id=self.options.camera_id,
            session_id=self.options.session_id,
            frame_store=frame_store,
        )

        publisher = AsyncProtobufKafkaPublisher(
            model_id=yolo.model_id,
            topic=topic,
            producer_config=producer_config_for_model(kafka_config, yolo.model_id),
            **publisher_options(kafka_config),
        )

        frame_queue: queue.Queue = queue.Queue(maxsize=self.options.queue_size)
        fatal_event = threading.Event()
        ready_event = threading.Event()
        sampler = FrameSampler(
            target_fps=self.options.sample_fps,
            fallback_source_fps=self.options.fallback_source_fps,
        )
        retriever_core = DeepStreamFrameRetrieverCore(
            cp=cp,
            gpu_id=self.options.gpu_id,
            frame_queue=frame_queue,
            sampler=sampler,
            fatal_event=fatal_event,
            max_samples=self.options.max_samples,
        )

        class ServiceMakerRetriever(BufferRetriever):
            def __init__(self, core: DeepStreamFrameRetrieverCore):
                super().__init__()
                self.core = core

            def consume(self, buffer: Any) -> int:
                return self.core.consume(buffer)

        retriever = ServiceMakerRetriever(retriever_core)
        metadata_probe = FrameMetadataProbe()
        metadata_probe_handle = Probe("frame-metadata", metadata_probe)
        worker = TensorRTEventWorker(
            settings=self.settings,
            options=self.options,
            frame_queue=frame_queue,
            publisher=publisher,
            frame_store=frame_store,
            manifest_writer=manifest_writer,
            ready_event=ready_event,
            fatal_event=fatal_event,
        )
        worker_thread = threading.Thread(
            target=worker.run,
            name="tensorrt-yolo11n-worker-01",
            daemon=False,
        )

        pipeline_error: BaseException | None = None
        publisher_error: str | None = None
        publisher_stats: dict[str, Any] = {}
        kafka_verification: dict[str, Any] = {
            "passed": False, "messages_received": 0
        }
        flow_seconds = delivery_seconds = verify_seconds = 0.0
        consumer_started = False
        publisher_started = False
        worker_started = False
        started = time.perf_counter()

        print("=" * 76)
        print("DEEPSTREAM + ONE CAMERA + YOLO11N TENSORRT FP16 + EVENT STORAGE + KAFKA")
        print("=" * 76)
        print(f"Source: {self.options.source}")
        print(f"Camera ID: {self.options.camera_id}")
        print(f"Session ID: {self.options.session_id}")
        print(f"Engine: {self.options.engine_path}")
        print(f"GPU: {self.options.gpu_id}")
        print(
            f"DeepStream batch: 1 x {self.options.batch_width} x "
            f"{self.options.batch_height}"
        )
        print(f"Sampling FPS: {self.options.sample_fps:.3f}")
        print(f"Sample interval: {1000.0 / self.options.sample_fps:.2f} ms")
        print(f"Fallback source FPS: {self.options.fallback_source_fps:.3f}")
        print(f"Event threshold: {self.options.event_min_confidence:.2f}")
        print(f"Event image root: {frame_store.root}")
        print(f"Kafka: {self.options.kafka_bootstrap_servers} / {topic}")
        print("Inference: TensorRT FP16 engine; no PyTorch/Ultralytics/OpenCV in container")

        try:
            verifier.wait_for_assignment()
            consumer_started = True

            publisher.start()
            publisher_started = True

            worker_thread.start()
            worker_started = True
            if not ready_event.wait(timeout=90):
                raise TimeoutError("TensorRT YOLO worker did not initialize")
            if worker.startup_errors:
                raise RuntimeError(
                    "TensorRT YOLO initialization failed: "
                    + "; ".join(worker.startup_errors)
                )

            print("[READY] DeepStream decoder, TensorRT worker and Kafka publisher are ready",
                  flush=True)
            flow = (
                Flow(Pipeline("one-camera-tensorrt-event-pipeline"))
                .capture([str(self.options.source)], gpu_id=self.options.gpu_id)
                # The batch stage creates NvDsBatchMeta/FrameMeta, including
                # frame_number, batch_id and source PTS used by the 8 FPS sampler.
                # For this one-camera pipeline, batch_size is intentionally 1.
                .batch(
                    batch_size=1,
                    width=self.options.batch_width,
                    height=self.options.batch_height,
                )
                .attach(metadata_probe_handle)
                .retrieve(retriever)
            )
            started = time.perf_counter()
            try:
                flow()
            finally:
                flow_seconds = time.perf_counter() - started

            if fatal_event.is_set():
                raise RuntimeError("A camera retriever or TensorRT worker failed")

        except BaseException as exc:
            pipeline_error = exc
            fatal_event.set()
            print(f"[PIPELINE ERROR] {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()

        finally:
            if worker_started and worker_thread.is_alive():
                frame_queue.put(None)
            if worker_started:
                frame_queue.join()
                worker_thread.join(timeout=120)
                if worker_thread.is_alive():
                    worker.worker_errors.append(
                        "TensorRT worker did not stop within 120 seconds"
                    )
                    fatal_event.set()

            if publisher_started:
                try:
                    publisher_stats = publisher.close()
                except Exception as exc:
                    publisher_error = f"{type(exc).__name__}: {exc}"
                    publisher_stats = publisher.stats()
                    print(f"[KAFKA SHUTDOWN ERROR] {publisher_error}", flush=True)
            else:
                publisher_stats = publisher.stats()

            delivery_seconds = time.perf_counter() - started
            if consumer_started:
                try:
                    expected_results = list(worker.results)
                    if expected_results and publisher_error is None:
                        verify_started = time.perf_counter()
                        try:
                            kafka_verification = verifier.verify(expected_results, self.options.verification_timeout)
                        finally:
                            verify_seconds = time.perf_counter() - verify_started
                    else:
                        kafka_verification["error"] = "No results were available to verify"
                except Exception as exc:
                    kafka_verification["error"] = f"{type(exc).__name__}: {exc}"
                    traceback.print_exc()
                finally:
                    consumer.close()
            else:
                consumer.close()

        elapsed_seconds = time.perf_counter() - started
        retriever_summary = retriever_core.summary()
        metadata_probe_summary = metadata_probe.get_summary()
        event_frames = sum(1 for item in worker.results if item["is_event"])
        detections_total = sum(
            int(item["detection_count"]) for item in worker.results
        )
        event_store_stats = frame_store.stats()

        completed = len(worker.results)
        rates = {
            "deepstream_decode": rate(retriever_summary["frames_seen"], flow_seconds, "frames", "flow to EOS"),
            "sampling": rate(retriever_summary["frames_sampled"], flow_seconds, "samples", "flow to EOS"),
            "kafka_delivery": rate(publisher_stats.get("delivered", 0), delivery_seconds, "messages", "flow start through publisher drain"),
            "kafka_verification": rate(kafka_verification.get("messages_received", 0), verify_seconds, "messages", "full verification pass including event files") if kafka_verification.get("passed") else None,
        }
        for stage in ("preprocess", "host_to_device", "tensorrt", "postprocess", "result_manifest", "event_storage"):
            count = event_frames if stage == "event_storage" else completed
            rates[stage] = rate(count, worker.active_ms[stage] / 1000, "events" if stage == "event_storage" else "samples", "summed stage time")

        summary = {
            "throughput": rates,
            "qualifying_frames": worker.qualifying_frames,
            "frames_cloned": retriever_summary["frames_cloned"],
            "schema_version": "1.0",
            "model_id": yolo.model_id,
            "engine": str(self.options.engine_path),
            "gpu_name": worker.detector.gpu_name if worker.detector else "unknown",
            "gpu_id": self.options.gpu_id,
            "source_video": str(self.options.source),
            "camera_id": self.options.camera_id,
            "source_session_id": self.options.session_id,
            "sample_fps_target": self.options.sample_fps,
            "sample_interval_ms": 1000.0 / self.options.sample_fps,
            "fallback_source_fps": self.options.fallback_source_fps,
            "frames_decoded": retriever_summary["frames_seen"],
            "metadata_probe": metadata_probe_summary,
            "frames_sampled": retriever_summary["frames_sampled"],
            "frames_skipped": retriever_summary["frames_skipped"],
            "frames_released": retriever_summary["frames_released"],
            "event_frames_saved": event_frames,
            "detections_total": detections_total,
            "average_detections_per_sampled_frame": (
                detections_total / len(worker.results) if worker.results else 0.0
            ),
            "event_min_confidence": self.options.event_min_confidence,
            "detection_class": self.options.detection_class,
            "event_every_nth_sample": self.options.event_every_nth_sample,
            "model_load_ms": worker.model_load_ms,
            "elapsed_seconds": elapsed_seconds,
            "end_to_end_sampled_fps": (
                len(worker.results) / elapsed_seconds if elapsed_seconds > 0 else 0.0
            ),
            "event_image_store": str(frame_store.root),
            "event_image_store_stats": event_store_stats,
            "results_manifest": str(manifest_path),
            "report_path": str(report_path),
            "kafka_topic": topic,
            "kafka_topic_status": topic_status,
            "kafka_publisher": publisher_stats,
            "kafka_verification": kafka_verification,
            "worker_errors": worker.startup_errors + worker.worker_errors,
            "retriever_errors": retriever_summary["errors"],
            "pipeline_error": (
                f"{type(pipeline_error).__name__}: {pipeline_error}"
                if pipeline_error else None
            ),
            "publisher_error": publisher_error,
        }

        checks = {
            "flow_completed": pipeline_error is None,
            "publisher_closed": publisher_error is None,
            "worker_initialized": not worker.startup_errors,
            "worker_completed": not worker.worker_errors,
            "retrieval_succeeded": retriever_summary["errors"] == 0,
            "metadata_matches_decode": metadata_probe_summary["total_frames"] == retriever_summary["frames_seen"],
            "received_samples": retriever_summary["frames_sampled"] > 0,
            "samples_completed": retriever_summary["frames_sampled"] == completed,
            "buffers_released": retriever_summary["frames_cloned"] == retriever_summary["frames_released"],
            "kafka_verified": kafka_verification.get("passed") is True,
        }
        for counter in ("enqueued", "produce_accepted", "delivered"):
            checks[f"kafka_{counter}"] = publisher_stats.get(counter, 0) == completed
        for counter in ("delivery_failures", "publish_failures", "shutdown_pending", "outstanding_after_produce", "queue_depth"):
            checks[f"kafka_{counter}_zero"] = publisher_stats.get(counter, 0) == 0
        failed_checks = [name for name, passed in checks.items() if not passed]
        summary["failed_acceptance_checks"] = failed_checks
        report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\n" + "=" * 76)
        print("TENSORRT EVENT PIPELINE SUMMARY")
        print("=" * 76)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print("=" * 76)
        if not failed_checks:
            print("DEEPSTREAM + YOLO11N TENSORRT + EVENT STORAGE + KAFKA TEST PASSED")
        else:
            print("DEEPSTREAM + YOLO11N TENSORRT + EVENT STORAGE + KAFKA TEST FAILED")
        print("=" * 76)

        if failed_checks:
            raise RuntimeError(f"Failed producer checks: {failed_checks}. See {report_path}")
        return summary


def _load_tensorrt_config(settings: AppSettings) -> dict[str, Any]:
    config_path = settings.paths.resolve(
        os.environ.get("APP_CONFIG_PATH") or settings.paths.config_file
    )
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    section = raw.get("tensorrt_pipeline", {})
    if not isinstance(section, dict):
        raise ValueError("tensorrt_pipeline must be a YAML mapping")
    return section


def build_parser(settings: AppSettings, raw: dict[str, Any]) -> argparse.ArgumentParser:
    model_id = settings.yolo.model_id
    default_topic = settings.kafka.topics.get(model_id, f"{model_id}-results")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(settings.yolo.source))
    parser.add_argument(
        "--camera-id", "--source-id", dest="camera_id",
        default=settings.source.camera_id,
    )
    parser.add_argument("--gpu-id", type=nonnegative_int, default=settings.yolo.gpu_id)
    parser.add_argument(
        "--sample-fps", type=float, default=float(raw.get("sample_fps", 8.0))
    )
    parser.add_argument(
        "--fallback-source-fps", type=float,
        default=float(raw.get("fallback_source_fps", 30.0)),
    )
    parser.add_argument(
        "--queue-size", type=positive_int,
        default=settings.queues.model_input_capacity,
    )
    parser.add_argument(
        "--batch-width",
        type=positive_int,
        default=positive_int(str(raw.get("batch_width", 1280))),
        help="DeepStream batched frame width; match source resolution to avoid resizing",
    )
    parser.add_argument(
        "--batch-height",
        type=positive_int,
        default=positive_int(str(raw.get("batch_height", 720))),
        help="DeepStream batched frame height; match source resolution to avoid resizing",
    )
    parser.add_argument(
        "--engine",
        default=str(settings.paths.resolve(raw.get(
            "engine", "artifacts/models/yolo11n_fp16.engine"
        ))),
    )
    parser.add_argument(
        "--event-min-confidence", type=float,
        default=settings.events.minimum_confidence,
    )
    parser.add_argument(
        "--detection-class", default=str(raw.get("detection_class", "person"))
    )
    parser.add_argument(
        "--event-every-nth-sample", type=positive_int,
        default=positive_int(str(raw.get("event_every_nth_sample", 1))),
    )
    parser.add_argument(
        "--event-type", default=str(raw.get("event_type", "person_detected"))
    )
    parser.add_argument("--topic", default=default_topic)
    parser.add_argument(
        "--kafka-bootstrap-servers",
        default=settings.kafka.bootstrap_servers,
    )
    parser.add_argument(
        "--result-root",
        default=str(settings.paths.resolve(raw.get(
            "result_root", "artifacts/results/yolo11n-person"
        ))),
    )
    parser.add_argument(
        "--jpeg-quality", type=positive_int,
        default=settings.events.jpeg_quality,
    )
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--max-samples", type=positive_int, default=None,
        help="Cap inference to the first N sampled frames; DeepStream still decodes to EOS",
    )
    parser.add_argument("--verification-timeout", type=float, default=90.0)
    return parser


def main(argv: list[str] | None = None, *, event_root: str) -> dict[str, Any]:
    settings = load_settings()
    raw = _load_tensorrt_config(settings)
    parser = build_parser(settings, raw)
    args = parser.parse_args(argv)

    options = TensorRTEventPipelineOptions(
        source=settings.paths.resolve(args.source),
        camera_id=args.camera_id,
        gpu_id=args.gpu_id,
        sample_fps=args.sample_fps,
        fallback_source_fps=args.fallback_source_fps,
        queue_size=args.queue_size,
        batch_width=args.batch_width,
        batch_height=args.batch_height,
        engine_path=settings.paths.resolve(args.engine),
        event_root=event_root,
        event_min_confidence=args.event_min_confidence,
        detection_class=args.detection_class,
        event_every_nth_sample=args.event_every_nth_sample,
        event_type=args.event_type,
        result_topic=args.topic,
        kafka_bootstrap_servers=args.kafka_bootstrap_servers,
        result_root=settings.paths.resolve(args.result_root),
        jpeg_quality=args.jpeg_quality,
        session_id=UUID(args.session_id).hex if args.session_id else uuid4().hex,
        max_samples=args.max_samples,
        verification_timeout=args.verification_timeout,
    )
    return TensorRTEventPipelineApplication(settings, options).run()
