# DeepStream AI Monitoring — one camera, one mock model

This branch is the first single-camera / single-model milestone. It intentionally
runs one MP4 source, one mock GPU model, and one Kafka result topic. It does not
yet add RTSP camera ingestion, real model weights, PostgreSQL persistence, or
event-image storage.

## Active test architecture

    One MP4 source (camera-01)
            |
    DeepStream decode (once)
            |
    One cloned GPU frame
            |
    One mock GPU model (model_a)
            |
    InferenceResult Protobuf adapter
            |
    One asynchronous Kafka publisher
            |
    model-a-results
            |
    Run-scoped Kafka consumer verifies every result

The default configuration enables model_a and disables model_b. The
configuration loader rejects zero or two enabled models. This is deliberate:
the active branch must have exactly one configured model.

Mock detections are synthetic and marked synthetic=true. This test validates
the GPU execution path, frame ownership, result contract, and Kafka delivery;
it does not measure real-world AI accuracy.

## Configuration

- configs/config.yaml: one enabled model, source, GPU, bounded input queue,
  Kafka publisher, and topic settings.
- configs/settings.py: typed dataclasses and validation.
- configs/paths.py: repository-root-aware pathlib paths.
- .env.example: local environment template. Copy it to .env if needed.
- src/pipeline/single_model_camera_test.py: active one-camera / one-model
  DeepStream + Kafka integration test.

The current test source is an MP4 file. The camera_id identifies the logical
camera in results; RTSP connection and reconnect behavior are a later milestone.

## Start Kafka and create the active topic

    docker compose up -d kafka

    docker compose exec kafka /opt/kafka/bin/kafka-topics.sh \
      --bootstrap-server localhost:9092 \
      --create --if-not-exists \
      --topic model-a-results \
      --partitions 3 --replication-factor 1

## Build the DeepStream container

    docker compose build dev
    docker compose run --rm dev nvidia-smi

## Run the one-model mock test

Use a video file that exists inside /workspace. For the local test video:

    docker compose run --rm dev \
      python3 -m src.pipeline.single_model_camera_test \
      --source /workspace/videos/TEST.mp4 \
      --source-id camera-01

The test:

1. Starts one run-scoped Kafka verifier on model-a-results.
2. Initializes one mock model on the selected GPU and one CUDA stream.
3. Decodes the video once through DeepStream.
4. Clones each frame once and submits it to the single model queue.
5. Runs synthetic GPU inference and builds the backend InferenceResult
   Protobuf message.
6. Publishes each result asynchronously to the one configured Kafka topic.
7. Flushes the publisher and consumes the messages for the current session.
8. Verifies frame sequence, camera ID, model ID, result IDs, timestamps,
   detections, and timing metadata.
9. Checks that every cloned GPU frame is released.

A successful run ends with:

    DEEPSTREAM + ONE CAMERA + ONE MOCK MODEL + KAFKA TEST PASSED

## CPU-only checks

    python -m compileall -q configs src
    python -m pytest -q tests/test_configuration.py \
      src/storage/event_frame_store_test.py \
      src/storage/verify_event_images_test.py \
      src/monitoring/metrics_test.py \
      src/pipeline/run_event_pipeline_test.py \
      src/results/protobuf_result_adapter_test.py

## Current boundaries

- The mock model generates synthetic detections; no trained weights are loaded.
- The initial camera source is an MP4 file, not RTSP.
- Only one enabled model and one worker are supported by this first test.
- Kafka publication is verified; PostgreSQL persistence and JPEG event storage
  are not connected to this standalone single-model test.
- The old two-model fan-out, event-persistence, and PostgreSQL end-to-end
  entry points are retained for reference but guarded against accidental use
  with this branch's one-model configuration.
- The GPU frame dispatcher uses a blocking bounded queue. Sustained overload
  can backpressure DeepStream decode.
- MP4 capture timestamps use DeepStream receiver wall-clock time, not the
  original recording timestamp.

Do not interpret a passing mock test as model-accuracy validation. The next
discussion can cover the real model interface and actual camera/RTSP source
after this one-model baseline is verified on the target GPU.
