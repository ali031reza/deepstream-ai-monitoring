"""
DeepStream single-camera shared-GPU-frame fan-out with real mock inference.

One DeepStream decode path extracts and clones each frame once. The same
CuPy GPU array is submitted to independent Model A and Model B workers.
Each worker owns a CUDA stream and a model instance. The shared frame is
kept alive until both workers release it.

This is a correctness-first, lossless fan-out test:
- Each model has a bounded, independent queue.
- Queue submission blocks when a queue is full (backpressure; no silent drops).
- Model inference results are materialized independently.
- The producer currently synchronizes after DLPack import to preserve the
  proven-safe ownership/readiness behavior from the previous fan-out test.
  This device-wide barrier is conservative and can limit cross-frame overlap.
"""

import argparse
import json
import math
import queue
import threading
import time
import traceback
from pathlib import Path

import cupy as cp

from pyservicemaker import BufferRetriever, Flow, Pipeline

from src.models.mock_gpu_model_a import MockGPUModelA
from src.models.mock_gpu_model_b import MockGPUModelB

from src.results.async_result_sink import AsyncJSONLResultSink
from src.results.model_result_message import build_result_message


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


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than zero.")
    return number


def nonnegative_int(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Value cannot be negative.")
    return number


def percentile(values, percent):
    if not values:
        return 0.0

    ordered = sorted(values)
    index = math.ceil((percent / 100.0) * len(ordered)) - 1
    index = max(0, min(index, len(ordered) - 1))
    return ordered[index]


class FrameTask:
    """Queue-specific dispatch metadata around one shared frame packet."""

    def __init__(self, packet):
        self.packet = packet
        self._enqueued_ns = None
        self._enqueued_event = threading.Event()

    def mark_enqueued(self):
        self._enqueued_ns = time.perf_counter_ns()
        self._enqueued_event.set()

    def queue_wait_ms(self):
        if not self._enqueued_event.wait(timeout=30):
            raise TimeoutError("Queue task was not marked as enqueued.")
        return (time.perf_counter_ns() - self._enqueued_ns) / 1_000_000.0


class SharedGPUFrame:
    """
    One GPU allocation shared by both model workers.

    Both the cloned DeepStream tensor and its CuPy view are retained until
    both consumers release their ownership reservations.
    """

    def __init__(self, frame_id, deepstream_tensor, gpu_array, on_release):
        self.frame_id = frame_id
        self._deepstream_tensor = deepstream_tensor
        self._gpu_array = gpu_array

        self.gpu_pointer = int(gpu_array.data.ptr)
        self.shape = tuple(gpu_array.shape)
        self.dtype = str(gpu_array.dtype)
        self.nbytes = int(gpu_array.nbytes)

        self._expected_consumers = set(CONSUMERS)
        self._released_consumers = set()
        self._lock = threading.Lock()
        self._fully_released = False
        self._on_release = on_release

    def get_gpu_array(self):
        with self._lock:
            if self._fully_released or self._gpu_array is None:
                raise RuntimeError(
                    f"Frame {self.frame_id} has already been released."
                )
            return self._gpu_array

    def release(self, consumer):
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
                # Clear packet ownership only after both workers have finished.
                self._gpu_array = None
                self._deepstream_tensor = None
                self._fully_released = True
                callback = self._on_release

        if callback is not None:
            callback()


class FanoutRetriever(BufferRetriever):
    """Clone each DeepStream frame once and submit the same packet to both queues."""

    def __init__(self, gpu_id, queue_a, queue_b, sample_limit=5):
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
        self.max_queue_depth = {"model_a": 0, "model_b": 0}
        self._stats_lock = threading.Lock()

    def _on_frame_released(self):
        with self._stats_lock:
            self.frames_fully_released += 1

    def consume(self, buffer):
        with self._stats_lock:
            frame_id = self.frames_seen
            self.frames_seen += 1

        raw_tensor = None
        owned_tensor = None
        gpu_array = None
        packet = None
        submitted_to = set()

        try:
            # 1. Extract the DeepStream tensor and clone it exactly once.
            raw_tensor = buffer.extract(0)
            if raw_tensor is None:
                raise RuntimeError("Failed to extract DeepStream tensor.")

            owned_tensor = raw_tensor.clone()
            del raw_tensor
            raw_tensor = None

            # 2. Import the cloned allocation into CuPy through DLPack.
            with cp.cuda.Device(self.gpu_id):
                device = tuple(owned_tensor.__dlpack_device__())
                expected_device = (CUDA_DLPACK_DEVICE_TYPE, self.gpu_id)

                if device != expected_device:
                    raise RuntimeError(
                        f"Unexpected DLPack device {device}; "
                        f"expected {expected_device}."
                    )

                gpu_array = cp.from_dlpack(owned_tensor)

                if gpu_array.device.id != self.gpu_id:
                    raise RuntimeError("CuPy array is on the wrong GPU.")

                if gpu_array.dtype != cp.uint8:
                    raise RuntimeError(
                        f"Unexpected frame dtype: {gpu_array.dtype}"
                    )

                if gpu_array.ndim != 3 or gpu_array.shape[2] != 3:
                    raise RuntimeError(
                        "Expected frame tensor with shape (H, W, 3); "
                        f"received {gpu_array.shape}."
                    )

                # Conservative readiness barrier retained from the previous
                # working fan-out implementation. See module docstring.
                cp.cuda.runtime.deviceSynchronize()

            # 3. Transfer ownership to the shared packet.
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

            # 4. Submit the same packet to each independent bounded queue.
            # Blocking puts deliberately preserve every frame in this test.
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
                print(f"Frame memory: {packet.nbytes / 1024**2:.3f} MiB")
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

            # Release reservations for queues that never received this packet.
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


def record_error(errors, errors_lock, model_id, message):
    with errors_lock:
        errors.append(f"{model_id}: {message}")


def drain_queue(
    frame_queue,
    model_id,
    worker_errors,
    worker_errors_lock,
):
    """
    Drain queued packets after worker initialization/fatal failure.

    Draining still releases this worker's reservation, allowing the other
    model to finish and preventing shared-frame retention.
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


def gpu_worker(
    model_id,
    frame_queue,
    gpu_id,
    model_config,
    source_id,
    result_sink,
    results,
    results_lock,
    worker_errors,
    worker_errors_lock,
    startup_errors,
    startup_errors_lock,
    ready_event,
    worker_info,
):
    """Own one model/stream and enqueue its independent result messages."""

    stream = None
    model = None
    initialized = False
    fatal_error = None

    try:
        init_start = time.perf_counter()

        with cp.cuda.Device(gpu_id):
            stream = cp.cuda.Stream(non_blocking=True)
            model = MODEL_CLASSES[model_id](
                cp=cp,
                gpu_id=gpu_id,
                stream=stream,
                **model_config,
            )
            stream.synchronize()

        initialization_ms = (time.perf_counter() - init_start) * 1000.0
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
            f"[{model_id}] Initialized {EXPECTED_RESULT_MODEL_IDS[model_id]} "
            f"on GPU {gpu_id}; CUDA stream={stream.ptr}; "
            f"init={initialization_ms:.2f} ms",
            flush=True,
        )

        while True:
            task = frame_queue.get()
            try:
                if task is None:
                    print(f"[{model_id}] Shutdown received.", flush=True)
                    return

                packet = task.packet
                frame = None
                release_attempted = False
                try:
                    queue_wait_ms = task.queue_wait_ms()
                    frame = packet.get_gpu_array()
                    frame_pointer = int(frame.data.ptr)
                    frame_object_id = id(frame)

                    processing_start = time.perf_counter_ns()
                    inference_result = model.infer(
                        frame,
                        frame_id=packet.frame_id,
                    )
                    # infer() materializes CPU results; this final barrier also
                    # guarantees that no stream operation still reads the frame.
                    stream.synchronize()
                    processing_ms = (
                        time.perf_counter_ns() - processing_start
                    ) / 1_000_000.0

                    # Inference has completed and its CUDA stream is synchronized.
                    # Release this worker's frame reservation before potentially
                    # blocking on the result queue, so slow disk/Kafka publishing
                    # cannot unnecessarily retain a GPU frame allocation.
                    if frame is not None:
                        del frame
                        frame = None
                    release_attempted = True
                    packet.release(model_id)

                    expected_result_id = EXPECTED_RESULT_MODEL_IDS[model_id]
                    if inference_result.get("model_id") != expected_result_id:
                        raise RuntimeError(
                            f"Unexpected result model_id "
                            f"{inference_result.get('model_id')!r}; "
                            f"expected {expected_result_id!r}."
                        )
                    if int(inference_result.get("frame_id", -1)) != packet.frame_id:
                        raise RuntimeError(
                            "Model result frame_id does not match input packet."
                        )

                    # Construct one self-contained envelope for this model only.
                    result_message = build_result_message(
                        inference_result,
                        source_id=source_id,
                        frame_shape=packet.shape,
                        queue_wait_ms=queue_wait_ms,
                        worker_processing_ms=processing_ms,
                    )

                    # This is a bounded asynchronous queue. If its writer falls
                    # behind, submit() applies backpressure instead of dropping.
                    result_sink.submit(result_message)

                    record = {
                        "frame_id": packet.frame_id,
                        "model_id": model_id,
                        "gpu_id": gpu_id,
                        "gpu_pointer": frame_pointer,
                        "array_object_id": frame_object_id,
                        "queue_wait_ms": queue_wait_ms,
                        "processing_ms": processing_ms,
                        "timings_ms": dict(inference_result.get("timings_ms", {})),
                        "detection_count": int(inference_result.get("detection_count", 0)),
                    }
                    with results_lock:
                        results[model_id].append(record)

                    if worker_info[model_id]["sample_message"] is None:
                        worker_info[model_id]["sample_message"] = result_message

                except Exception as exc:
                    record_error(
                        worker_errors,
                        worker_errors_lock,
                        model_id,
                        f"Frame {packet.frame_id}: {exc}",
                    )
                    traceback.print_exc()

                    # A failed sink cannot recover by retrying every subsequent
                    # frame. Stop this worker and drain its input queue safely.
                    if result_sink.error is not None or not result_sink.is_alive:
                        raise RuntimeError(
                            f"Result sink for {model_id} is unavailable."
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


def print_worker_statistics(model_id, records, worker_info):
    queue_waits = [item["queue_wait_ms"] for item in records]
    processing_times = [item["processing_ms"] for item in records]
    model_timings = [item["timings_ms"] for item in records]

    def average(key):
        values = [item.get(key, 0.0) for item in model_timings]
        return sum(values) / len(values) if values else 0.0

    print(f"\n========== {model_id.upper()} ==========")
    print(f"Model result ID: {EXPECTED_RESULT_MODEL_IDS[model_id]}")
    print(f"Completed inference results: {len(records)}")
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

    info = worker_info.get(model_id, {})
    print(f"CUDA stream pointer: {info.get('stream_pointer', 'unavailable')}")
    print(f"Model initialization: {info.get('initialization_ms', 0.0):.3f} ms")

    sample_message = info.get("sample_message")
    if sample_message is not None:
        print(f"Sample detection count: {sample_message.get('detection_count', 0)}")
        print("Sample result envelope (first detection only):")
        sample_view = dict(sample_message)
        sample_view["detections"] = sample_message.get("detections", [])[:1]
        print(json.dumps(sample_view, indent=2))

    print("================================")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "DeepStream shared GPU frame fan-out, two mock models, and "
            "independent asynchronous result sinks"
        )
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-id", default="camera_0")
    parser.add_argument("--gpu-id", type=nonnegative_int, default=0)
    parser.add_argument("--queue-size", type=positive_int, default=4)
    parser.add_argument("--sample-limit", type=nonnegative_int, default=5)

    parser.add_argument("--output-dir", default="artifacts/results")
    parser.add_argument("--result-queue-size", type=positive_int, default=128)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--flush-interval-ms", type=positive_int, default=100)

    parser.add_argument("--model-a-input-size", type=positive_int, default=320)
    parser.add_argument("--model-a-hidden-units", type=positive_int, default=128)
    parser.add_argument("--model-a-top-k", type=positive_int, default=5)
    parser.add_argument("--model-a-seed", type=int, default=42)

    parser.add_argument("--model-b-input-size", type=positive_int, default=320)
    parser.add_argument("--model-b-hidden-units", type=positive_int, default=512)
    parser.add_argument("--model-b-top-k", type=positive_int, default=5)
    parser.add_argument("--model-b-seed", type=int, default=84)
    return parser


def main():
    args = build_parser().parse_args()

    video_path = Path(args.source)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not args.source_id.strip():
        raise ValueError("--source-id must not be empty.")

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

    output_dir = Path(args.output_dir)
    result_paths = {
        "model_a": output_dir / "model_a_results.jsonl",
        "model_b": output_dir / "model_b_results.jsonl",
    }

    print("=" * 72)
    print("DEEPSTREAM TWO-MODEL INFERENCE + ASYNCHRONOUS RESULT SINKS")
    print("=" * 72)
    print(f"Video: {video_path}")
    print(f"Source ID: {args.source_id}")
    print(f"GPU: {gpu_name}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Model A config: input={args.model_a_input_size}, "
          f"hidden={args.model_a_hidden_units}, top_k={args.model_a_top_k}")
    print(f"Model B config: input={args.model_b_input_size}, "
          f"hidden={args.model_b_hidden_units}, top_k={args.model_b_top_k}")
    print(f"Input queue capacity per model: {args.queue_size}")
    print(f"Result queue capacity per model: {args.result_queue_size}")
    print(f"Result batch size: {args.batch_size}")
    print(f"Result flush interval: {args.flush_interval_ms} ms")
    print("Publishing mode: independent asynchronous JSONL sinks (Kafka stand-in)")
    print("Output files are overwritten at the beginning of each run.")

    queues = {
        "model_a": queue.Queue(maxsize=args.queue_size),
        "model_b": queue.Queue(maxsize=args.queue_size),
    }
    results = {"model_a": [], "model_b": []}
    results_lock = threading.Lock()

    worker_errors = []
    worker_errors_lock = threading.Lock()
    startup_errors = []
    startup_errors_lock = threading.Lock()
    worker_info = {}

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

    result_sinks = {
        "model_a": AsyncJSONLResultSink(
            model_id=EXPECTED_RESULT_MODEL_IDS["model_a"],
            output_path=result_paths["model_a"],
            queue_size=args.result_queue_size,
            batch_size=args.batch_size,
            flush_interval_ms=args.flush_interval_ms,
        ),
        "model_b": AsyncJSONLResultSink(
            model_id=EXPECTED_RESULT_MODEL_IDS["model_b"],
            output_path=result_paths["model_b"],
            queue_size=args.result_queue_size,
            batch_size=args.batch_size,
            flush_interval_ms=args.flush_interval_ms,
        ),
    }

    ready_events = {model_id: threading.Event() for model_id in CONSUMERS}
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
                result_sinks[model_id],
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

    retriever = FanoutRetriever(
        gpu_id=args.gpu_id,
        queue_a=queues["model_a"],
        queue_b=queues["model_b"],
        sample_limit=args.sample_limit,
    )

    started_sinks = set()
    started_workers = set()
    pipeline_error = None
    pipeline_elapsed_seconds = 0.0

    try:
        print("\nStarting independent asynchronous result sinks...", flush=True)
        for model_id in CONSUMERS:
            result_sinks[model_id].start()
            started_sinks.add(model_id)

        print("Starting independent GPU model workers...", flush=True)
        for model_id in CONSUMERS:
            threads[model_id].start()
            started_workers.add(model_id)

        for model_id in CONSUMERS:
            if not ready_events[model_id].wait(timeout=60):
                raise TimeoutError(
                    f"{model_id} initialization did not finish within 60 seconds."
                )

        if startup_errors:
            raise RuntimeError(f"GPU worker initialization failed: {startup_errors}")

        print("\nBoth GPU workers and both result sinks are ready.", flush=True)
        print("Building DeepStream pipeline...", flush=True)

        pipeline_start = time.perf_counter()
        flow = (
            Flow(Pipeline("shared-gpu-frame-result-publishing"))
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
        print("\nSending model-worker shutdown signals...", flush=True)
        for model_id in started_workers:
            queues[model_id].put(None)

        for model_id in started_workers:
            queues[model_id].join()
        for model_id in started_workers:
            threads[model_id].join()

        print("Closing result sinks and flushing final batches...", flush=True)
        for model_id in started_sinks:
            try:
                result_sinks[model_id].close()
            except Exception as exc:
                worker_errors.append(
                    f"{model_id} result sink shutdown: {type(exc).__name__}: {exc}"
                )

    summary = retriever.get_summary()
    sink_stats = {model_id: result_sinks[model_id].stats for model_id in CONSUMERS}
    sink_errors = [
        f"{model_id}: {sink_stats[model_id]['error']}"
        for model_id in CONSUMERS
        if sink_stats[model_id]["error"] is not None
    ]

    print("\n" + "=" * 72)
    print("DEEPSTREAM + TWO-MODEL + ASYNC RESULT PUBLISHING RESULTS")
    print("=" * 72)
    print(f"Frames received: {summary['frames_seen']}")
    print(f"GPU frame clones created: {summary['frames_cloned']}")
    print(f"Frames submitted to both models: {summary['frames_submitted']}")
    print(f"Frames fully released: {summary['frames_fully_released']}")
    print(f"Producer errors: {summary['errors']}")
    print(f"Model A inference results: {len(results['model_a'])}")
    print(f"Model B inference results: {len(results['model_b'])}")
    print(f"Pipeline elapsed: {pipeline_elapsed_seconds:.3f} seconds")

    throughput = (
        summary["frames_submitted"] / pipeline_elapsed_seconds
        if pipeline_elapsed_seconds > 0 else 0.0
    )
    print(f"Submitted-frame throughput: {throughput:.2f} FPS")
    print(f"Model A maximum input queue depth: {summary['max_queue_depth_a']}")
    print(f"Model B maximum input queue depth: {summary['max_queue_depth_b']}")

    for model_id in CONSUMERS:
        print_worker_statistics(model_id, results[model_id], worker_info)

    print("\n========== INDEPENDENT RESULT SINKS ==========")
    for model_id in CONSUMERS:
        stats = sink_stats[model_id]
        print(f"\n{model_id.upper()}")
        print(f"Model ID: {stats['model_id']}")
        print(f"Messages enqueued: {stats['enqueued_messages']}")
        print(f"Messages written: {stats['written_messages']}")
        print(f"Batches written: {stats['batches_written']}")
        print(f"Maximum batch size: {stats['max_batch_size']}")
        print(f"Maximum result-queue depth: {stats['max_queue_depth']}")
        print(f"Submissions that encountered backpressure: {stats['blocked_submissions']}")
        print(f"Average result-queue submit wait: "
              f"{(stats['submit_wait_total_ms'] / stats['enqueued_messages']) if stats['enqueued_messages'] else 0.0:.3f} ms")
        print(f"Maximum result-queue submit wait: {stats['submit_wait_max_ms']:.3f} ms")
        print(f"Output: {stats['output_path']}")
        print(f"Sink error: {stats['error']}")

    model_a_by_frame = {item["frame_id"]: item for item in results["model_a"]}
    model_b_by_frame = {item["frame_id"]: item for item in results["model_b"]}
    expected_frames = set(range(summary["frames_submitted"]))
    model_a_frames = set(model_a_by_frame)
    model_b_frames = set(model_b_by_frame)
    matching_frames = (
        model_a_frames == expected_frames and model_b_frames == expected_frames
    )

    shared_pointer_matches = 0
    shared_object_matches = 0
    for frame_id in model_a_frames & model_b_frames:
        result_a = model_a_by_frame[frame_id]
        result_b = model_b_by_frame[frame_id]
        if result_a["gpu_pointer"] == result_b["gpu_pointer"]:
            shared_pointer_matches += 1
        if result_a["array_object_id"] == result_b["array_object_id"]:
            shared_object_matches += 1

    streams_are_independent = (
        worker_info.get("model_a", {}).get("stream_pointer") is not None
        and worker_info.get("model_b", {}).get("stream_pointer") is not None
        and worker_info["model_a"]["stream_pointer"]
        != worker_info["model_b"]["stream_pointer"]
    )

    print("\n" + "=" * 72)
    print("ARCHITECTURE + RESULT DELIVERY VALIDATION")
    print("=" * 72)
    print(f"Matching frame IDs for both models: {matching_frames}")
    print(
        f"Shared GPU pointer matches: "
        f"{shared_pointer_matches}/{summary['frames_submitted']}"
    )
    print(
        f"Shared CuPy object matches: "
        f"{shared_object_matches}/{summary['frames_submitted']}"
    )
    print(
        "All cloned frames fully released: "
        f"{summary['frames_fully_released'] == summary['frames_cloned']}"
    )
    print(f"Independent CUDA streams: {streams_are_independent}")
    print(
        "Model A sink wrote every result: "
        f"{sink_stats['model_a']['written_messages'] == summary['frames_submitted']}"
    )
    print(
        "Model B sink wrote every result: "
        f"{sink_stats['model_b']['written_messages'] == summary['frames_submitted']}"
    )
    print(f"Model outputs remain separate: {result_paths['model_a'] != result_paths['model_b']}")
    print(f"Producer errors: {summary['errors']}")
    print(f"Worker errors: {len(worker_errors)}")
    print(f"Sink errors: {len(sink_errors)}")
    print(f"Startup errors: {len(startup_errors)}")

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
        sink_stats["model_a"]["enqueued_messages"] == summary["frames_submitted"],
        sink_stats["model_b"]["enqueued_messages"] == summary["frames_submitted"],
        sink_stats["model_a"]["written_messages"] == summary["frames_submitted"],
        sink_stats["model_b"]["written_messages"] == summary["frames_submitted"],
        len(worker_errors) == 0,
        len(sink_errors) == 0,
        len(startup_errors) == 0,
        streams_are_independent,
    ])

    print("\n" + "=" * 72)
    if successful:
        print("TWO-MODEL INFERENCE + INDEPENDENT ASYNC RESULT SINK TEST PASSED")
    else:
        print("TWO-MODEL INFERENCE + INDEPENDENT ASYNC RESULT SINK TEST FAILED")
        for error in worker_errors:
            print(f"WORKER ERROR: {error}")
        for error in sink_errors:
            print(f"SINK ERROR: {error}")
        for error in startup_errors:
            print(f"STARTUP ERROR: {error}")
        if pipeline_error is not None:
            print(f"PIPELINE ERROR: {pipeline_error}")
    print("=" * 72)

    if not successful:
        raise RuntimeError("Two-model inference/result publishing validation failed.")


if __name__ == "__main__":
    main()
