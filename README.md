# DeepStream AI Monitoring — one camera, one model

This branch maintains the validated one-camera / one-model DeepStream mock
pipeline and adds a standalone pretrained YOLO11n person-detection experiment.
The standalone YOLO test runs in the WSL Python environment; the existing
DeepStream Docker image remains unchanged until its PyTorch/TensorRT runtime
compatibility is verified.

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
executes on the CUDA GPU. It does not yet replace the mock model inside
DeepStream. The current dev container has CuPy but does not have PyTorch,
Torchvision, Ultralytics, OpenCV, or the TensorRT Python module. Do not add
PyTorch blindly to the DeepStream image. First verify the container's CUDA,
Python, and TensorRT compatibility, then choose the production integration
path (PyTorch in-container or TensorRT/DeepStream inference).

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

- The mock DeepStream pipeline still uses synthetic detections.
- YOLO11n is a standalone real-model video test, not yet connected to
  DeepStream frame retrieval or Kafka.
- The camera source is an MP4 file; RTSP support is a later milestone.
- PostgreSQL persistence and event-image storage are not connected to the
  standalone YOLO runner.
- The old two-model fan-out, event-persistence, and PostgreSQL end-to-end
  entry points remain reference workflows and are guarded against accidental
  use with this branch's one-model configuration.
