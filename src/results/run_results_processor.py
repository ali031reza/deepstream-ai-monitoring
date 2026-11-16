from __future__ import annotations

import argparse
import logging
import signal
import threading

from src.kafka.consumer import KafkaResultConsumer
from src.kafka.runtime_config import load_runtime_config
from src.kafka.dead_letter_publisher import DeadLetterPublisher
from src.kafka.async_batch_writer import AsyncBatchWriter
from src.database.client import DatabaseClient
from src.database.batch_result_repository import DatabaseBatchRepository
from src.results.results_processing_service import ResultsProcessingService


def main():
    parser = argparse.ArgumentParser(
        description="DeepStream Kafka Results Processor"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-messages", type=int, default=None)
    args = parser.parse_args()

    if args.max_messages is not None and args.max_messages <= 0:
        parser.error("--max-messages must be positive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    config = load_runtime_config(args.config)
    stop_event = threading.Event()

    def request_shutdown(signum, frame):
        logging.warning("Shutdown signal received: %s", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)

    database = DatabaseClient(**config["database"])
    if not database.check_connection():
        raise RuntimeError("PostgreSQL connection check failed")

    repository = DatabaseBatchRepository(database)

    consumer = KafkaResultConsumer(config["consumer_config"])

    dead_letter = DeadLetterPublisher(
        bootstrap_servers=config["bootstrap_servers"],
        topic=config["dead_letter_topic"],
        producer_config=config["dead_letter_producer_config"],
        flush_timeout_seconds=config[
            "dead_letter_flush_timeout_seconds"
        ],
    )

    batch = config["batch"]
    writer = AsyncBatchWriter(
        persist_batch=repository.persist_batch,
        dead_letter_publisher=dead_letter,
        batch_size=batch["batch_size"],
        flush_interval_ms=batch["flush_interval_ms"],
        queue_capacity=batch["queue_capacity"],
        retry_attempts=batch["retry_attempts"],
        retry_backoff_ms=batch["retry_backoff_ms"],
    )

    service = ResultsProcessingService(
        consumer=consumer,
        batch_writer=writer,
        dead_letter_publisher=dead_letter,
        topic_model_map=config["topic_model_map"],
        poll_timeout_seconds=config[
            "consumer_config"
        ].poll_timeout_seconds,
        shutdown_timeout_seconds=batch[
            "shutdown_timeout_seconds"
        ],
    )

    logging.info(
        "Starting processor topics=%s batch_size=%d flush_interval_ms=%d",
        list(config["topic_model_map"]),
        batch["batch_size"],
        batch["flush_interval_ms"],
    )

    service.run(
        stop_event=stop_event,
        max_messages=args.max_messages,
    )


if __name__ == "__main__":
    main()
