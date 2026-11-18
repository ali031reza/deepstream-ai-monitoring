from __future__ import annotations

import logging
import threading
import time

from src.kafka.records import KafkaRecord
from src.kafka.protobuf_codec import deserialize_inference_result
from src.kafka.message_validator import validate_inference_result


logger = logging.getLogger(__name__)


class BatchProcessingError(RuntimeError):
    """Raised when a background batch cannot be safely acknowledged."""


class ResultsProcessingService:
    """
    Coordinates Kafka polling/commits on the main thread and PostgreSQL/DLT
    work on the asynchronous batch-writer thread.
    """

    def __init__(
        self,
        *,
        consumer,
        batch_writer,
        dead_letter_publisher,
        topic_model_map,
        poll_timeout_seconds=0.5,
        shutdown_timeout_seconds=180.0,
    ):
        if not topic_model_map:
            raise ValueError("No result topics configured")

        self.consumer = consumer
        self.batch_writer = batch_writer
        self.dead_letter_publisher = dead_letter_publisher
        self.topic_model_map = dict(topic_model_map)
        self.allowed_model_ids = set(topic_model_map.values())
        self.poll_timeout_seconds = poll_timeout_seconds
        self.shutdown_timeout_seconds = shutdown_timeout_seconds

        self._metrics = {
            "prepare_seconds": 0.0,
            "received": 0,
            "submitted": 0,
            "processed": 0,
            "dead_lettered": 0,
            "batches": 0,
            "commit_failures": 0,
            "commit_time_ms": 0.0,
            "backpressure_waits": 0,
            "rebalance_fences": 0,
        }

    def _prepare_record(self, message):
        record = KafkaRecord.from_message(message).with_assignment_epoch(
            getattr(self.consumer, "rebalance_epoch", None)
        )

        try:
            result = deserialize_inference_result(record.payload)
            record = record.with_result(result)

            expected_model = self.topic_model_map.get(record.topic)
            if expected_model is None:
                raise ValueError(f"Unexpected Kafka topic: {record.topic}")
            if result.model_id != expected_model:
                raise ValueError(
                    f"Model/topic mismatch: expected={expected_model}, "
                    f"received={result.model_id}"
                )

            validate_inference_result(
                result,
                allowed_model_ids=self.allowed_model_ids,
                require_schema_version=True,
            )
            return record
        except Exception as exc:
            logger.warning(
                "Invalid Kafka result topic=%s partition=%s offset=%s: %s",
                record.topic,
                record.partition,
                record.offset,
                exc,
            )
            return record.with_error(exc)

    def _ensure_fresh_assignment(self, record):
        record_epoch = getattr(record, "assignment_epoch", None)
        current_epoch = getattr(self.consumer, "rebalance_epoch", None)
        if (
            record_epoch is not None
            and current_epoch is not None
            and record_epoch != current_epoch
        ):
            self._metrics["rebalance_fences"] += 1
            raise BatchProcessingError(
                "Partition assignment changed with an uncommitted Kafka "
                "record; stop and replay rather than process stale ownership"
            )

    def _service_paused_consumer(self):
        # Unlike time.sleep(), polling while partitions are paused keeps
        # callbacks and group membership serviced. Unexpected messages fail
        # closed rather than being silently discarded.
        self.consumer.pause()
        self.consumer.service_events(0.05)

    def _drain_writer(self):
        deadline = time.monotonic() + self.shutdown_timeout_seconds
        while self.batch_writer.is_alive():
            self._drain_completions()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Database batch writer did not stop before timeout"
                )
            self._service_paused_consumer()
            self.batch_writer.join(min(0.05, remaining))
        self._drain_completions()

    def _drain_completions(self):
        while True:
            completion = self.batch_writer.get_completion_nowait()
            if completion is None:
                break

            if not completion.success:
                raise BatchProcessingError(
                    completion.error or "Unknown batch processing failure"
                )

            for record in completion.records:
                self._ensure_fresh_assignment(record)

            commit_started = time.perf_counter()
            try:
                # DB transaction and all DLT deliveries completed in worker.
                # Commit only here, on the Kafka consumer's owning thread.
                self.consumer.commit_batch(completion.records)
            except Exception:
                self._metrics["commit_failures"] += 1
                raise
            finally:
                self._metrics["commit_time_ms"] += (
                    time.perf_counter() - commit_started
                ) * 1000.0

            self._metrics["batches"] += 1
            self._metrics["processed"] += len(completion.records) - completion.dead_letter_count
            self._metrics["dead_lettered"] += completion.dead_letter_count

            logger.info(
                "Batch acknowledged total=%d persisted=%d dlt=%d",
                len(completion.records),
                completion.persisted_count,
                completion.dead_letter_count,
            )

    def run(
        self,
        *,
        stop_event: threading.Event | None = None,
        max_messages: int | None = None,
    ):
        if max_messages is not None and max_messages <= 0:
            raise ValueError("max_messages must be positive or None")
        if stop_event is None:
            stop_event = threading.Event()

        submitted = 0
        pending_record = None
        paused = False
        failure = None
        writer_started = False

        try:
            self.consumer.subscribe()
            self.batch_writer.start()
            writer_started = True

            logger.info(
                "Results processor started topics=%s",
                list(self.topic_model_map),
            )

            while (
                not stop_event.is_set()
                and (max_messages is None or submitted < max_messages)
            ):
                self._drain_completions()

                if self.batch_writer.fatal_error:
                    raise BatchProcessingError(
                        str(self.batch_writer.fatal_error)
                    )

                if pending_record is not None:
                    self._ensure_fresh_assignment(pending_record)
                    if self.batch_writer.submit(pending_record):
                        submitted += 1
                        self._metrics["submitted"] += 1
                        pending_record = None
                        if paused:
                            self.consumer.resume()
                            paused = False
                    else:
                        paused = True
                        self._metrics["backpressure_waits"] += 1
                        self._service_paused_consumer()
                    continue

                if self.batch_writer.queue_full:
                    paused = True
                    self._metrics["backpressure_waits"] += 1
                    self._service_paused_consumer()
                    continue

                if paused:
                    self.consumer.resume()
                    paused = False

                message = self.consumer.poll(self.poll_timeout_seconds)
                if message is None:
                    continue

                prepare_start = time.perf_counter()
                record = self._prepare_record(message)
                self._metrics["prepare_seconds"] += time.perf_counter() - prepare_start
                self._metrics["received"] += 1
                if self.batch_writer.submit(record):
                    submitted += 1
                    self._metrics["submitted"] += 1
                else:
                    # Keep the already-polled record; never drop or commit it.
                    pending_record = record
                    self.consumer.pause()
                    paused = True

            # Submit the one locally held record before draining on shutdown.
            while pending_record is not None:
                self._drain_completions()
                if self.batch_writer.fatal_error:
                    raise BatchProcessingError(
                        str(self.batch_writer.fatal_error)
                    )
                self._ensure_fresh_assignment(pending_record)
                if self.batch_writer.submit(pending_record):
                    submitted += 1
                    self._metrics["submitted"] += 1
                    pending_record = None
                else:
                    self._metrics["backpressure_waits"] += 1
                    self._service_paused_consumer()

            self.consumer.pause()
            self.batch_writer.request_stop()
            self._drain_writer()
            if self.batch_writer.fatal_error:
                raise BatchProcessingError(
                    str(self.batch_writer.fatal_error)
                )

        except Exception as exc:
            failure = exc
            logger.exception("Results processing failed")

        finally:
            if writer_started:
                self.batch_writer.request_stop()
                if self.batch_writer.is_alive():
                    try:
                        self._drain_writer()
                    except Exception as exc:
                        failure = failure or exc
                        # On an already-failed consumer, still allow the
                        # worker to terminate without committing any stale
                        # batches during a blocking fallback join.
                        self.batch_writer.join(self.shutdown_timeout_seconds)

                try:
                    self._drain_completions()
                except Exception as exc:
                    failure = failure or exc

            try:
                self.dead_letter_publisher.close()
            except Exception as exc:
                failure = failure or exc

            self.consumer.close()

            logger.info(
                "Results processor stopped metrics=%s writer_stats=%s",
                self._metrics,
                self.batch_writer.stats(),
            )

        if failure is not None:
            raise failure

        return dict(self._metrics)
