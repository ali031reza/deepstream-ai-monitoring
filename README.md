# Standalone Kafka result publishing layer

This package adds a new Kafka publishing layer to `deepstream-ai-monitoring`.
It does not import code from the separate `KAFKA_AI_Monitoring` repository.

## Architecture

- One `AsyncKafkaPublisher` instance for `mock_model_a`.
- One `AsyncKafkaPublisher` instance for `mock_model_b`.
- Each publisher owns its own bounded Python queue and Kafka producer.
- Each model writes to its own Kafka topic.
- Messages use a versioned JSON envelope.
- `confluent-kafka` performs asynchronous delivery and native producer batching.
- The publisher worker drains up to `max_batch_messages` per application batch.
- `linger.ms` and `batch.size` control librdkafka's native batching.

The publishers never combine Model A and Model B results.

## Files

- `config/kafka.json`: broker, topic, queue and producer settings.
- `schemas/model_result.schema.json`: documented JSON message contract.
- `src/results/model_result_message.py`: validated result envelope.
- `src/results/kafka_config.py`: configuration loading and environment overrides.
- `src/results/async_kafka_publisher.py`: independent asynchronous publisher.
- `src/results/kafka_publisher_test.py`: real Kafka publish/consume integration test.
- `requirements.txt`: preserves PyYAML and adds confluent-kafka.

## Install

Copy the files into the matching paths in the project, replacing
`requirements.txt` with the supplied complete file.

Rebuild the development image:

```bash
docker compose build dev
```

The existing Dockerfile already installs `requirements.txt`; no Dockerfile
change is required.

## Broker connectivity

Inside the Docker Compose network, the default bootstrap address is:

```text
kafka:9092
```

The Windows host address is not used by the `dev` container.

## Run the integration test

Make sure the Kafka service is healthy, then run:

```bash
docker compose run --rm dev   python3 -m src.results.kafka_publisher_test   --messages-per-model 100   --consume-timeout 60
```

The test creates unique temporary topics, publishes 100 messages per model,
consumes them, validates model identity, frame IDs, message IDs and payload
contents, then deletes the temporary topics.

To retain the temporary topics for inspection, add:

```text
--keep-test-topics
```

## Production topics

The configured production topics are:

- `model-a-results`
- `model-b-results`

Create them after the broker is healthy:

```bash
docker compose exec kafka   /opt/kafka/bin/kafka-topics.sh   --bootstrap-server localhost:9092   --create --if-not-exists   --topic model-a-results   --partitions 3   --replication-factor 1

docker compose exec kafka   /opt/kafka/bin/kafka-topics.sh   --bootstrap-server localhost:9092   --create --if-not-exists   --topic model-b-results   --partitions 3   --replication-factor 1
```

## Important delivery semantics

`enable.idempotence=true` and `acks=all` improve producer-side delivery
reliability. They do not make the whole inference-to-Kafka pipeline exactly
once. If the application retries an inference result after an uncertain
shutdown, downstream consumers should use `message_id` for deduplication.

The Python result queue is bounded. If it fills, `publish()` applies
backpressure for the configured enqueue timeout and then raises `TimeoutError`
rather than silently dropping a result.

This first phase is intentionally independent of the DeepStream frame fan-out
pipeline. The next integration step will convert each completed model result
to `ModelResultMessage` and enqueue it to that model's publisher.
