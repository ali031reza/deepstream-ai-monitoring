# DeepStream end-to-end reliability test

Run one local video from its first frame to end of stream. DeepStream samples decoded frames, YOLO11n identifies people, and each sampled frame produces one Kafka message and one PostgreSQL result. Qualifying event frames also produce a JPEG and a JSON sidecar. A session report reconciles these outputs and measures throughput at each stage.

## Run the Paris walk

From the project root in WSL, use the existing video at `videos/reliability/paris_city_walk.mp4`, GPU-compatible engine at `artifacts/models/yolo11n_fp16.engine`, and project `.env`. The defaults match the 1920 × 1080, 60 FPS source and sample it at 8 FPS. Docker Compose needs NVIDIA GPU access. PostgreSQL runs outside this Compose project. It must be reachable with the application's credentials and have the existing base schema, event_images table and DeepStream metadata migration applied. The runner checks the schema before decoding.

```bash
docker compose up -d --wait kafka
docker compose run --rm -T dev python3 -u -m src.pipeline.run_tensorrt_end_to_end \
  --source /workspace/videos/reliability/paris_city_walk.mp4 \
  --camera-id camera-01 --sample-fps 8 --verification-timeout 600
```

A zero exit code and the final PASS confirm a completed run. For a different clip, set `--batch-width` and `--batch-height` to its dimensions. Sampling uses presentation timestamps. Set `--fallback-source-fps` to the source frame rate for frames with missing timestamps. Omit `--max-samples` for full coverage. Compare the report's decoded count with the source video frame count. Each invocation generates a new session ID.

For a fresh environment, copy `.env.example` to `.env`, supply the PostgreSQL application password and connection settings, and run `docker compose build dev`. If the local engine needs rebuilding, place YOLO11n weights in `artifacts/models/yolo11n.pt`, run `python -m tools.export_yolo11n_onnx` in a Python environment with `requirements-yolo.txt` installed, then run `docker compose run --rm --no-deps dev python3 -m tools.build_yolo11n_tensorrt_engine` on the target GPU. A compatible existing engine skips these preparation steps.

## Outputs

| Output under the project directory | Contents |
|---|---|
| `artifacts/event_frames/events/<camera-hash>/<session>/<frame-sequence>.jpg` | Decoded event image at the configured batch resolution |
| JSON beside each JPEG | Source timestamp, detection boxes and confidence, event identity, image size and SHA-256 |
| `artifacts/results/yolo11n-person/<session>/results.jsonl` | One line per sampled frame, including source frame sequence, number of qualifying detections, event flag and image key |
| `artifacts/results/yolo11n-person/<session>/performance.json` | Producer counts, throughput, publisher counters and verifier diagnostics |
| `artifacts/benchmarks/processor_<session>.json` | Drained Kafka consumer and PostgreSQL writer counters |
| `artifacts/benchmarks/tensorrt_end_to_end_<session>.json` | Final frame counts, stage throughput, verified database rows and unique images |

The frame summary distinguishes **decoded**, **sampled**, **skipped**, **qualifying**, **event_frames**, **images_written**, **images_reused** and **verified_images**. A qualifying frame has at least one person detection above the configured event threshold (default 0.65, inclusive). It counts once even when several people appear. The event policy can save every Nth sample if it qualifies. The default N is 1. Each manifest entry links a stored frame to its source sequence and PTS.

## Interpreting throughput

Every measurement includes a count, seconds, unit and timing window. Average throughput is total completed items divided by total measured seconds. A zero-length window yields a null rate. Stages overlap, and the bounded input queue can slow capture while inference or disk I/O catches up.

| Stage | Unit and timing interval |
|---|---|
| DeepStream decode and sampling | Decoded frames/s and selected samples/s over the same capture-to-EOS wall time, including backpressure |
| Preprocessing | Samples/s over summed preparation time, including copying the selected GPU frame to the host and letterboxing |
| Host-to-device transfer and TensorRT | Separate samples/s over the respective summed CUDA event times |
| Postprocessing | Samples/s over summed output processing time |
| Result manifest | Sample records/s over JSONL append calls |
| Event storage | Event pairs/s over summed JPEG store and JSON write calls, including any reuse |
| Kafka delivery | Acknowledged messages/s from flow start through producer drain |
| Kafka verification | Verified messages/s during independent replay, including event-image and sidecar checks |
| Consumer validation | Messages/s over deserialization and validation in the results processor |
| PostgreSQL writes | Persisted rows/s over summed batch persistence time, including retries |
| End to end | Verified session rows/s from processor launch through database/image verification and clean shutdown |

Use the end-to-end rate to compare runs with the same source, sampling policy and storage location. The 8 FPS setting uses the video clock. Reported throughput uses processing time. The consumer and writer metrics cover their processor instance, so use a quiet result topic for isolated measurements.

## Structure and design decisions

| Location | Responsibility |
|---|---|
| `configs/`, `docker/`, `docker-compose.yml` | Typed settings, DeepStream 9.2 container, NVIDIA runtime and Kafka broker |
| `src/pipeline/run_tensorrt_end_to_end.py` | Public launcher. Creates a fresh session and Kafka group, waits for assignment, runs the producer, reconciles results, shuts down the processor and writes the report |
| `src/pipeline/`, `src/models/` | DeepStream Service Maker and CuPy retrieve frames. A PTS sampler selects frames before cloning GPU memory. A bounded queue hands selected frames to one TensorRT FP16 worker and applies backpressure |
| `src/storage/` | Pillow encodes event JPEGs. Atomic writes and deterministic camera/session/sequence keys keep event files associated with their results |
| `schemas/`, `src/proto/`, `src/results/` | Protobuf contract and generated bindings, async publisher and consumer lifecycle. Every sampled frame carries a result, including those with no event |
| `src/kafka/`, `src/database/` | Kafka validation, batched psycopg writes, dead-letter handling and offset commits following completed batches |
| `infra/postgres/`, `tools/` | Metadata migration for the provisioned database and optional ONNX export / TensorRT engine preparation |

The independent verifier checks the session's Kafka sequences, event flags and referenced files. The supervisor checks PostgreSQL result and event counts and verifies unique JPEG hashes. If the producer fails acceptance, inspect `failed_acceptance_checks` in its `performance.json` and process exit code. `--verification-timeout` controls the Kafka and event-file verification pass. `--db-timeout` separately controls the wait for PostgreSQL convergence.
