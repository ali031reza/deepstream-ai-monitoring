# DeepStream AI Monitoring — single-camera, two-model POC

This repository contains the standalone DeepStream proof of concept for
Kafka AI Monitoring. It does not import code from the separate
`KAFKA_AI_Monitoring` repository.

## Active DeepStream architecture

The current end-to-end path is:

```text
One MP4 source
    |
DeepStream decode
    |
One shared GPU frame allocation
    |
    +--------------------------+
    |                          |
Model A GPU worker        Model B GPU worker
    |                          |
InferenceResult adapter   InferenceResult adapter
    |                          |
Async Protobuf publisher  Async Protobuf publisher
    |                          |
model-a-results           model-b-results
```

Important properties:

- The source is decoded once.
- Both models receive the same shared GPU frame allocation.
- Each model has its own bounded input queue, worker, publisher queue,
  Kafka producer, and result topic.
- Each model publishes independently; neither waits for the other.
- Kafka carries binary Protobuf inference results, not raw video frames.
- A unique `source_session_id` is generated for each application run.
- The result ID is deterministic for a camera/session/frame/model pair.
- The pipeline test consumes the results after producer flush and verifies
  the actual binary Protobuf messages from that run.

The event-pipeline runner supports independent configurable worker pools for
both mock models. Configure the defaults in configs/config.yaml or override via CLI.

## Protobuf result contract

The active DeepStream path uses the backend `InferenceResult` Protobuf
message defined in `schemas/inference_result.proto`.

The adapter maps:

- `camera_id`
- `source_session_id`
- `frame_id` and `frame_sequence`
- `model_id` and `worker_id`
- `result_id`
- `captured_at_ms` and `processed_at_ms`
- detections, normalized bounding boxes, confidence values, and synthetic
  flags
- GPU and pipeline stage timings
- event-image metadata when an image has actually been stored

Mock outputs are synthetic; the event pipeline can select and persist qualifying
frames, but the mock models do not establish real-world detection quality.
The pipeline must not publish a successful event reference before its
image has been stored.

For the MP4 proof of concept, `captured_at_ms` records the wall-clock time
when DeepStream hands the frame to the retriever. This is an arrival-time
fallback, not the original recording time. Source PTS/NTP timestamp
handling must be finalized when the source moves to RTSP.

## Kafka configuration

`configs/config.yaml` defines the broker, model-specific topics, bounded
publisher queues, batching, and producer settings.

Inside the Docker Compose network, the bootstrap address is:

```text
kafka:9092
```

The `dev` container uses this internal address. The Windows host address
is not used from inside the container.

Configured production topics:

- `model-a-results`
- `model-b-results`

The development broker disables automatic topic creation. Create the
topics after Kafka is healthy:

```bash
docker compose up -d kafka

docker compose exec kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --if-not-exists \
  --topic model-a-results \
  --partitions 3 --replication-factor 1

docker compose exec kafka /opt/kafka/bin/kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --create --if-not-exists \
  --topic model-b-results \
  --partitions 3 --replication-factor 1
```

## Files

- `configs/config.yaml`: broker, topic, queue, batching, and producer
  settings.
- `schemas/inference_result.proto`: backend Protobuf contract.
- `src/proto/inference_result_pb2.py`: generated Python Protobuf module.
- `src/results/protobuf_result_adapter.py`: converts model output to
  `InferenceResult`.
- `src/results/async_protobuf_kafka_publisher.py`: active binary
  Protobuf publisher, one instance per model.
- `src/results/protobuf_result_adapter_test.py`: Protobuf adapter tests.
- `src/results/protobuf_kafka_publisher_test.py`: standalone binary
  Protobuf Kafka publish/consume test.
- `src/pipeline/shared_gpu_frame_fanout_kafka_test.py`: DeepStream
  shared-GPU fan-out, model execution, binary Protobuf publishing, and
  actual Kafka consumer verification.
- `src/results/async_kafka_publisher.py` and
  `src/results/model_result_message.py`: retained legacy JSON publishing
  path for compatibility and isolated tests; the active DeepStream
  pipeline no longer uses them.
- `src/results/kafka_publisher_test.py`: legacy JSON publisher test.

## Build the development image

The Dockerfile installs the Python dependencies from `requirements.txt`:

```bash
docker compose build dev
```

Rebuild only when the Dockerfile or Python dependencies change. Source
files are bind-mounted into `/workspace`.

## Run the Protobuf adapter test

```bash
docker compose run --rm dev \
  python3 -m src.results.protobuf_result_adapter_test
```

## Run the standalone Protobuf Kafka test

Make sure Kafka is healthy and the development container can reach
`kafka:9092`, then run:

```bash
docker compose run --rm dev \
  python3 -m src.results.protobuf_kafka_publisher_test \
  --messages-per-model 100 \
  --consume-timeout 60
```

This test creates unique temporary topics, publishes binary Protobuf
messages for both models, consumes and validates them, then deletes the
temporary topics. Add `--keep-test-topics` to retain them.

## Run the full DeepStream + Protobuf pipeline

Use a video file that is available inside the container under
`/workspace`. Replace the example path with the actual MP4 path:

```bash
docker compose run --rm dev bash -lc '
set -e

python3 -m py_compile \
  src/pipeline/shared_gpu_frame_fanout_kafka_test.py

python3 -m src.pipeline.shared_gpu_frame_fanout_kafka_test \
  --source /workspace/videos/sample.mp4 \
  --source-id camera-01
'
```

The `--source` path must exist inside the container. If the video is in
the repository's `videos` directory, it will normally be available at
`/workspace/videos/...`.

The pipeline test:

1. Subscribes a run-specific Kafka consumer to both result topics and
   initializes its offsets before publishing begins.
2. Starts one independent Protobuf publisher per model.
3. Starts the two GPU model workers.
4. Decodes the MP4 once and fans the shared GPU frame out to both models.
5. Publishes each model's `InferenceResult` independently.
6. Flushes both publishers on shutdown.
7. Consumes the binary Protobuf messages for the current
   `source_session_id`.
8. Verifies message counts, model/topic identity, frame identity,
   timestamps, detections, and pipeline timing metadata.

A successful run ends with:

```text
DEEPSTREAM + TWO GPU MODELS + INDEPENDENT BINARY PROTOBUF KAFKA TEST PASSED
```

## Delivery semantics and current limitations

- `acks=all` and `enable.idempotence=true` improve producer-side
  delivery reliability. They do not make the complete inference-to-
  database workflow exactly once.
- Downstream consumers should deduplicate using `result_id`.
- Publisher queues are bounded. If a queue remains full beyond its
  configured enqueue timeout, publishing fails visibly rather than
  silently dropping a result.
- The MP4 test currently uses the DeepStream receiver wall-clock time for
  `captured_at_ms`; it does not claim this is the source recording time.
- Event-image storage and PostgreSQL persistence are not yet connected to
  this standalone DeepStream pipeline.
- Configurable worker pools, RTSP ingestion, and real AI models remain
  later milestones.

## Central configuration and environment

All non-secret defaults live in **`configs/config.yaml`**. Typed dataclasses
and validation live in `configs/settings.py`; `configs/paths.py` resolves
relative paths from the repository root rather than the current directory.
Kafka publishing, result consumption, PostgreSQL connection settings, source,
mock models, queues, event-frame storage and benchmark defaults are configured
there. Explicit legacy JSON configuration files are still accepted by the
Kafka adapters, but no JSON copy is shipped.

Create a local environment file before running services:

```bash
cp .env.example .env
# Edit DATABASE_PASSWORD (and any overrides) locally.
```

`.env` is excluded from Git and Docker build context. **Never commit
credentials.** Environment variables supplied by the shell or Docker override
values in `.env` and the YAML Kafka/database defaults. Use
`APP_CONFIG_PATH` or `PROJECT_ROOT` for nonstandard deployments.

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest tests/test_configuration.py
python -m src.results.run_results_processor --config configs/config.yaml
python -m src.pipeline.run_end_to_end --source /workspace/videos/your-video.mp4
```

DeepStream/GPU commands require the NVIDIA DeepStream environment; the config
unit tests do not. Generated events, reports and JSONL result sinks belong under
`artifacts/` and are ignored by Git.
