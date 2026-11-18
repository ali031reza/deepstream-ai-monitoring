# DeepStream AI Monitoring — one camera, one model

This branch maintains the validated one-camera / one-model DeepStream mock
pipeline, the standalone YOLO11n reference experiment, and a separate YOLO11n
TensorRT FP16 event pipeline. The TensorRT runner uses the DeepStream 9.2
container and keeps PyTorch/Ultralytics out of that runtime.

## Active DeepStream mock architecture

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
configuration loader rejects zero or two enabled mock models. This remains
the validated DeepStream/Kafka baseline.

## Standalone real-model experiment: YOLO11n

The branch also contains a separate real-model video test:

    videos/TEST.mp4
            |
    Ultralytics YOLO11n (pretrained COCO weights)
            |
    CUDA GPU 0, FP16, 640 x 640
            |
    COCO class 0 only (person)
            |
    Annotated video + performance.json

The default settings are in the yolo section of configs/config.yaml.
Weights are downloaded automatically to artifacts/models/yolo11n.pt.
Annotated videos and performance reports are written under
artifacts/videos/. The entire artifacts/ directory is ignored by Git;
model weights and generated media are not committed.

### Install the tested WSL YOLO environment

Use the existing dedicated virtual environment:

    source "$HOME/.venvs/yolo11n/bin/activate"
    cd /mnt/d/projects/deepstream-ai-monitoring
    python -m pip install -r requirements-yolo.txt

The requirements file pins the versions verified on the RTX 3050:
PyTorch 2.14.1+cu132, Torchvision 0.29.1+cu132, and Ultralytics 8.4.215.

### Run person detection

    python -m src.pipeline.run_yolo11n_video

The source defaults to videos/TEST.mp4. To select another video:

    python -m src.pipeline.run_yolo11n_video --source /absolute/path/to/video.mp4

Expected output:

- Model weights: artifacts/models/yolo11n.pt
- Annotated video: artifacts/videos/yolo11n-person-detection/
- Performance report: artifacts/videos/yolo11n-person-detection/performance.json

The report records processed frames, total detections, model-load time,
preprocess/inference/postprocess timings, end-to-end FPS, inference FPS,
GPU identity, and peak CUDA memory.

### Important boundary

This standalone test uses Ultralytics/PyTorch and confirms that YOLO11n
executes on the CUDA GPU. The TensorRT event runner is a separate real-model
path: it exports ONNX in the WSL YOLO environment, builds a TensorRT FP16
engine in the matching DeepStream container, and runs that engine from the
DeepStream frame-retrieval worker. It does not replace the mock baseline.

## Existing DeepStream/Kafka mock test

Start Kafka and create the active result topic:

    docker compose up -d kafka

    docker compose exec kafka /opt/kafka/bin/kafka-topics.sh \
      --bootstrap-server localhost:9092 \
      --create --if-not-exists \
      --topic model-a-results \
      --partitions 3 --replication-factor 1

Build the DeepStream container:

    docker compose build dev
    docker compose run --rm dev nvidia-smi

Run the one-camera / one-mock-model test:

    docker compose run --rm dev \
      python3 -m src.pipeline.single_model_camera_test \
      --source /workspace/videos/TEST.mp4 \
      --source-id camera-01

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

The YOLO test itself requires the optional GPU dependencies in
requirements-yolo.txt and is not part of CPU-only CI.

## Current boundaries

- The original mock DeepStream pipeline still uses synthetic detections.
- The YOLO11n TensorRT runner is a separate one-camera real-model path and
  publishes Protobuf results to Kafka after writing qualifying JPEG/JSON files.
- The existing results processor can persist the new model's results to
  PostgreSQL when its schema/migration and database connection are available.
- The first integrated source is an MP4 file; RTSP reconnects and camera-origin
  wall-clock timestamps remain later work.
- The old two-model fan-out and event-persistence runners remain reference
  workflows; this branch's active mock baseline still enables one model.

## YOLO11n TensorRT event pipeline (one camera)

This adds a separate real-model runner while leaving the existing mock
DeepStream/Kafka runner intact.

    MP4 / one camera
        |
    DeepStream decode once
        |
    PTS sampler (8 frames/second; 125 ms target interval)
        |
    YOLO11n TensorRT FP16 engine (COCO person class 0)
        |
    event filter (event_frames.minimum_confidence)
        |
        +--> JPEG + JSON sidecar (only qualifying person frames)
        |
        +--> Protobuf result --> yolo11n-person-results
                                  |
                                  +--> existing ResultsProcessingService
                                       --> PostgreSQL

### What is saved

For every sampled frame, the run writes one JSONL record under
`artifacts/results/yolo11n-person/<session-id>/results.jsonl`. This is an
inference manifest, not an image archive.

For every meaningful frame (person confidence at or above
`event_frames.minimum_confidence`), the pipeline writes:

- A JPEG of the original decoded frame, without drawn boxes.
- A JSON sidecar with `event_type`, `detection.type`, confidence and
  bounding boxes, `camera_id`, model/engine identity, source frame sequence,
  sample index, timestamps, and image size/SHA-256.

The JPEG and sidecar share the same storage key stem and live below
`artifacts/event_frames/events/<camera-hash>/<session-id>/`. The Kafka
Protobuf message contains the detections and the image storage reference;
it does not carry JPEG bytes.

Timestamp meanings are deliberately separate:

- `source_video_timestamp_ms`: source presentation timestamp (PTS).
- `captured_at_ms`: UTC wall-clock time when this process observed the
  decoded frame. For an MP4, this is not the original camera-recording time.
- `processed_at_ms`: UTC wall-clock time after inference.

The sampler uses PTS, not "every eighth decoded frame". At 8 FPS the target
boundaries are 125 ms apart; for a 30 FPS file it selects the first decoded
frame at or after each boundary.

### 1. Export ONNX in the existing YOLO environment

Run this on the WSL host using the already documented YOLO virtualenv.
This is the only stage that uses Ultralytics/PyTorch:

    source "$HOME/.venvs/yolo11n/bin/activate"
    cd /mnt/d/projects/deepstream-ai-monitoring
    python -m pip install -r requirements-yolo.txt
    python -m tools.export_yolo11n_onnx

Expected artifact:

    artifacts/models/yolo11n.onnx

### 2. Build the matching DeepStream/TensorRT image

DeepStream 9.2 is based on CUDA 13.2 and TensorRT 10.16.0.72. The DeepStream base image supplies the native TensorRT runtime, ONNX parser,
and `trtexec`. The Dockerfile installs the matching TensorRT 10.16.0.72
Python bindings from NVIDIA's CUDA 13 PyPI package, avoiding an APT update
against the base image's external CUDA repository.

    docker compose build dev
    docker compose run --rm dev nvidia-smi
    docker compose run --rm dev python3 -c "import tensorrt as trt; print(trt.__version__)"

Build the engine inside this container on the target RTX 3050. TensorRT
engines are tied to their TensorRT runtime and target GPU architecture:

    docker compose run --rm dev \
      python3 -m tools.build_yolo11n_tensorrt_engine

Expected artifact:

    artifacts/models/yolo11n_fp16.engine

### 3. Start Kafka and run the inference pipeline

    docker compose up -d kafka

Run a smoke test with inference capped at 120 sampled frames. For a genuinely
short run, use a source clip of about 15 seconds; the cap limits inference,
but DeepStream still decodes the source through end-of-stream:

    docker compose run --rm dev \
      python3 -m src.pipeline.run_tensorrt_event_pipeline \
      --source /workspace/videos/TEST.mp4 \
      --camera-id camera-01 \
      --sample-fps 8 \
      --max-samples 120

For a full video run, omit `--max-samples 120`. Kafka topic
`yolo11n-person-results` is created/verified automatically by the runner.
The runner verifies that every sampled frame result was delivered and that
each event message references an existing JPEG and JSON sidecar.

### 4. Optional: persist Kafka results into PostgreSQL

The existing `src.results.run_results_processor` consumes all configured
result topics, validates Protobuf messages, and persists inference metadata
and event-image references using `DatabaseBatchRepository`. Because
`yolo11n-person` is now in the Kafka topic map, the existing processor
includes this model without a second consumer implementation.

First ensure the existing PostgreSQL schema and migration
`infra/postgres/migrations/003_deepstream_result_metadata.sql` are applied,
and configure the database password in `.env`. Then start the processor in
a second terminal:

    docker compose run --rm dev \
      python3 -m src.results.run_results_processor

The database processor uses the configured `DATABASE_HOST`,
`DATABASE_PORT`, `DATABASE_NAME`, `DATABASE_USER`, and password
environment variables. The current compose file starts Kafka, not PostgreSQL;
a reachable PostgreSQL instance is still required for this optional stage.

### 5. CPU-only tests and syntax checks

    python -m compileall -q configs src tools
    python -m pytest -q \
      tests/test_frame_sampler.py \
      tests/test_event_metadata_store.py \
      tests/test_tensorrt_yolo_output_decoder.py \
      tests/test_tensorrt_result_contract.py

These tests cover timestamp-based 8 FPS sampling, event JPEG/JSON sidecar
integrity, and YOLO output decoding/NMS. They do not execute CUDA, TensorRT,
DeepStream decoding, or Kafka; the container smoke test above is required
for the full hardware integration check.

### Runtime notes

- `yolo.confidence` controls which detections enter the result.
  `event_frames.minimum_confidence` independently controls whether a
  sampled frame is saved as a meaningful event. `tensorrt_pipeline.event_every_nth_sample`
  controls image-write decimation without reducing inference sampling FPS.
- The current event rule is the configured detection class (default `person`,
  COCO class 0) meeting the event threshold. `tensorrt_pipeline.event_every_nth_sample`
  defaults to `1`, so every qualifying sampled frame is saved. Increase it to
  decimate event-image writes; no cross-frame tracking is implemented.
- The worker copies each sampled GPU frame to host memory for PIL
  letterboxing, then transfers the model input to TensorRT. TensorRT
  inference itself runs on the GPU. This is a simple, testable first
  integration; optimize preprocessing/zero-copy only after measuring it.
- The MP4 path is the first target. RTSP reconnects and camera-origin
  wall-clock timestamps remain later work.
