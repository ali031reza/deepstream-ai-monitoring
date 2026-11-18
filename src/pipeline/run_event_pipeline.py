"""Phases 2/3/4: extend the validated DeepStream/CuPy two-model runner.

This runner REUSES src.pipeline.shared_gpu_frame_fanout_kafka_test instead of
implementing a second GPU decode path. It wraps model.infer while the GPU frame
is still alive; event JPEG persistence finishes before protobuf publication.

Example:
  python3 -m src.pipeline.run_event_pipeline --source /workspace/video.mp4 \
    --source-id camera_0 --model-a-workers 2 --model-b-workers 3
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import queue as std_queue
import sys
import threading
import time
from types import SimpleNamespace
from uuid import UUID, uuid4

from configs.settings import AppSettings, load_settings
from src.monitoring.metrics import RunMetrics, save_json
from src.monitoring.resource_sampler import ResourceSampler
from src.storage.event_frame_store import EventFrameStore
from src.storage.event_policy import EventPolicy


def _base_option(args: list[str], flag: str, default: str):
    for idx, item in enumerate(args):
        if item == flag and idx + 1 < len(args):
            return args[idx + 1]
        if item.startswith(flag + "="):
            return item[len(flag) + 1:]
    return default


def make_event_model(original_type, *, model_key, camera_id, session_id,
                     store, policy, metrics, enabled=True):
    """Model wrapper guarantees image exists before Kafka sees event metadata."""
    class EventModel:
        def __init__(self, *args, **kwargs):
            self._model = original_type(*args, **kwargs)

        def infer(self, frame, *, frame_id):
            started = time.perf_counter()
            output = self._model.infer(frame, frame_id=frame_id)
            metrics.add(f"{model_key}.infer", duration_ms=(time.perf_counter()-started)*1000)
            output = dict(output)
            # The mock models do NOT perform real-world event recognition.
            qualifies = enabled and policy.matches(output, frame_id)
            output["is_event"] = qualifies
            if qualifies:
                started = time.perf_counter()
                image = store.put_rgb(
                    camera_id=camera_id, source_session_id=session_id,
                    frame_sequence=frame_id, rgb_array=frame,
                )
                output.update({
                    "event_type": "synthetic_detection",
                    "storage_key": image.storage_key,
                    "image_format": image.image_format,
                    "image_size_bytes": image.image_size_bytes,
                    "image_sha256": image.image_sha256,
                })
                metrics.add(f"{model_key}.event", duration_ms=(time.perf_counter()-started)*1000)
            return output

        def __getattr__(self, name):
            return getattr(self._model, name)
    return EventModel


def make_publisher_proxy(publisher_type, metrics):
    class PublisherProxy:
        def __init__(self, *args, **kwargs):
            self._real = publisher_type(*args, **kwargs)
            self._model_id = kwargs.get("model_id", "unknown")

        def publish(self, message, **kwargs):
            start = time.perf_counter()
            result = self._real.publish(message, **kwargs)
            metrics.add(f"{self._model_id}.kafka_enqueue", duration_ms=(time.perf_counter()-start)*1000)
            return result

        def __getattr__(self, name):
            return getattr(self._real, name)
    return PublisherProxy


class MultiSentinelQueue(std_queue.Queue):
    """One shutdown sentinel from the existing runner wakes *all* workers."""
    def __init__(self, *, maxsize: int, workers: int):
        super().__init__(maxsize=maxsize)
        if workers < 1:
            raise ValueError("workers must be >= 1")
        self.workers = workers

    def put(self, item, block=True, timeout=None):
        if item is None:
            # Queue.join accounts for each additional sentinel.
            for _ in range(self.workers):
                super().put(None, block=block, timeout=timeout)
        else:
            super().put(item, block=block, timeout=timeout)


def make_queue_factory(workers_a: int, workers_b: int):
    created = []
    def new_queue(*, maxsize=0):
        # Original pipeline only creates model_a and model_b queues here.
        index = len(created)
        if index > 1:
            raise RuntimeError("Unexpected queue allocation by pipeline")
        q = MultiSentinelQueue(maxsize=maxsize, workers=(workers_a, workers_b)[index])
        created.append(q)
        return q
    return SimpleNamespace(Queue=new_queue, Empty=std_queue.Empty, Full=std_queue.Full)


def make_worker_supervisor(original_gpu_worker, workers_a, workers_b):
    def run_workers(*args):
        model_key = args[0]
        n = workers_a if model_key == "model_a" else workers_b
        main_ready = args[13]
        # Only signal the pipeline ready after every worker finished init.
        ready_events = [threading.Event() for _ in range(n)]
        children = []
        for i in range(1, n):
            copy_args = list(args)
            copy_args[13] = ready_events[i]
            th = threading.Thread(
                target=original_gpu_worker, args=tuple(copy_args),
                name=f"{model_key}-worker-{i+1:02d}", daemon=False,
            )
            th.start()
            children.append(th)

        def signal_all_ready():
            for event in ready_events:
                event.wait()
            main_ready.set()
        watcher = threading.Thread(target=signal_all_ready, daemon=True)
        watcher.start()
        local_args = list(args)
        local_args[13] = ready_events[0]
        try:
            original_gpu_worker(*local_args)
        finally:
            for child in children:
                child.join()
            watcher.join(timeout=1)
            main_ready.set()
    return run_workers


def pin_verifier_start_offsets(consumer, assignment, partition_type):
    """Avoid `latest` consumer group startup races in the existing verifier.

    Capture each source topic's high-watermark before publishers begin. The
    verifier is a disposable consumer group specific to this DeepStream run.
    """
    starts = []
    for part in assignment:
        topic_part = partition_type(part.topic, part.partition)
        low, high = consumer.get_watermark_offsets(topic_part, timeout=10, cached=False)
        starts.append(partition_type(part.topic, part.partition, high))
    consumer.assign(starts)
    return starts


def make_safe_verifier_assignment(original_wait, partition_type):
    def safe_wait(consumer, timeout_seconds=30):
        assignment = original_wait(consumer, timeout_seconds=timeout_seconds)
        pin_verifier_start_offsets(consumer, assignment, partition_type)
        return assignment
    return safe_wait


@dataclass(slots=True)
class EventPipelineApplication:
    """Configure the event-pipeline runner and own its full run lifecycle."""

    settings: AppSettings

    def run(self, argv=None):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--event-root", default=str(self.settings.events.root))
        parser.add_argument("--event-min-confidence", type=float, default=self.settings.events.minimum_confidence)
        parser.add_argument("--event-every-n", type=int, default=self.settings.events.every_nth_frame)
        parser.add_argument("--event-class", default=self.settings.events.class_name)
        parser.add_argument("--jpeg-quality", type=int, default=self.settings.events.jpeg_quality)
        parser.add_argument("--model-a-workers", type=int, default=self.settings.models["model_a"].workers)
        parser.add_argument("--model-b-workers", type=int, default=self.settings.models["model_b"].workers)
        parser.add_argument("--session-id", default=None)
        parser.add_argument("--metrics-json", default=None)
        parser.add_argument("--resource-interval", type=float, default=self.settings.resource_interval_seconds)
        opts, original_args = parser.parse_known_args(argv)
        if opts.model_a_workers < 1 or opts.model_b_workers < 1:
            parser.error("Worker counts must be >= 1")
        if opts.resource_interval <= 0:
            parser.error("--resource-interval must be > 0")
        session = UUID(opts.session_id).hex if opts.session_id else uuid4().hex
        camera_id = _base_option(original_args, "--source-id", self.settings.source.camera_id)
        store = EventFrameStore(self.settings.paths.resolve(opts.event_root), quality=opts.jpeg_quality)
        policy = EventPolicy(opts.event_min_confidence, opts.event_every_n, opts.event_class)
        metrics = RunMetrics()
        sampler = ResourceSampler(interval_seconds=opts.resource_interval).start()
        metrics_target = opts.metrics_json or self.settings.metrics_directory / f"phase234_{session}.json"
        failure = None
        print(f"PHASE234 source_session_id={session}", flush=True)
        print(f"PHASE234 event_root={store.root} model_a_workers={opts.model_a_workers} "
              f"model_b_workers={opts.model_b_workers}", flush=True)
        try:
            # Heavy CuPy/pyservicemaker imports happen only inside the DeepStream container.
            from src.pipeline import shared_gpu_frame_fanout_kafka_test as pipeline

            original_models = dict(pipeline.MODEL_CLASSES)
            pipeline.MODEL_CLASSES = {
                key: make_event_model(cls, model_key=key, camera_id=camera_id,
                                      session_id=session, store=store,
                                      policy=policy, metrics=metrics,
                                      enabled=self.settings.events.enabled)
                for key, cls in original_models.items()
            }
            pipeline.AsyncProtobufKafkaPublisher = make_publisher_proxy(
                pipeline.AsyncProtobufKafkaPublisher, metrics,
            )
            original_adapter = pipeline.build_inference_result
            def with_worker_identity(*args, **kwargs):
                kwargs["worker_id"] = threading.current_thread().name.replace("_", "-")
                return original_adapter(*args, **kwargs)
            pipeline.build_inference_result = with_worker_identity
            original_worker = pipeline.gpu_worker
            pipeline.gpu_worker = make_worker_supervisor(
                original_worker, opts.model_a_workers, opts.model_b_workers,
            )
            pipeline.queue = make_queue_factory(opts.model_a_workers, opts.model_b_workers)
            pipeline.uuid4 = lambda: UUID(hex=session)
            # The original Kafka verifier subscribes using `latest`. A single
            # poll after assignment is insufficient to guarantee resolved start
            # offsets; explicitly pin its start to the pre-publish high-watermarks.
            from confluent_kafka import TopicPartition
            pipeline.wait_for_consumer_assignment = make_safe_verifier_assignment(
                pipeline.wait_for_consumer_assignment, TopicPartition,
            )
            old_argv = sys.argv
            try:
                sys.argv = [old_argv[0], *original_args]
                pipeline.main()
            finally:
                sys.argv = old_argv
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            metrics.error(failure)
            raise
        finally:
            samples = sampler.stop()
            report = {
                "schema": "phase234-run-v1", "source_session_id": session,
                "camera_id": camera_id,
                "event_store": str(store.root), "event_store_stats": store.stats(),
                "event_policy": vars(opts).copy(),
                "worker_counts": {"model_a": opts.model_a_workers, "model_b": opts.model_b_workers},
                "metrics": metrics.snapshot(), "resources": samples,
                "failed": failure,
            }
            save_json(self.settings.paths.resolve(metrics_target), report)
            print(f"PHASE234 metrics_json={self.settings.paths.resolve(metrics_target)}", flush=True)
            print(f"PHASE234 event_store_stats={json.dumps(store.stats())}", flush=True)



def main(argv=None):
    settings = load_settings()
    if sum(model.enabled for model in settings.models.values()) != 2:
        raise SystemExit(
            "The event-persistence runner is still a two-model workflow; "
            "use src.pipeline.single_model_camera_test for this branch's first milestone"
        )
    EventPipelineApplication(settings).run(argv)


if __name__ == "__main__":
    main()
