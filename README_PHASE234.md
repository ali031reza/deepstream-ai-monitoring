> Historical phase guide. For current configuration and setup, see README.md and configs/config.yaml.

# DeepStream AI Monitoring — Phases 2, 3 and 4

**Target:** `ali031reza/deepstream-ai-monitoring`, branch `DeepStream_Pipeline`, with the **Phase 1 v2 Kafka consumer files already installed locally**.

**Package type:** **additive patch**. It includes only new files. It does **not** replace your existing working DeepStream fan-out implementation, Kafka consumer, Protobuf schema, PostgreSQL migration, or Docker Compose file. No GitHub updates were made.

## Features

### Phase 2 — Persistent event-frame JPEG storage

- `src/storage/event_policy.py` selects qualifying **mock** detections; default confidence >= 0.65, sampled every 30th frame. This is not real object detection.
- `src/storage/event_frame_store.py` copies the **original shared GPU frame** to CPU only for qualifying events, JPEG-encodes it, and atomically writes it to `artifacts/event_frames/`.
- Both model workers produce the exact same storage key for the same `(camera_id, source_session_id, frame_sequence)`. A mutex and SHA-256 check ensure a single image file with no silent collisions.
- **The JPEG write occurs inside `model.infer` before `packet.release(...)` and before Kafka publishes a result referencing that image.** This avoids publishing a reference to a file that has not yet been stored.
- `src/storage/verify_event_images.py` checks PostgreSQL image metadata against actual file size and SHA-256, without modifying data.
- Event image *bytes* do **not** enter Kafka: only Protobuf storage key, size, SHA, format and event information do. The existing Phase 1 PostgreSQL repository persists the event image row and foreign key.

### Phase 3 — Fully connected local workflow

- `src/pipeline/run_event_pipeline.py` **reuses** your validated `src/pipeline/shared_gpu_frame_fanout_kafka_test.py` without duplicating DeepStream decoding, GPU frame ownership or independent per-model Kafka topics.
- Adds independent Model A/B worker counts (`--model-a-workers`, `--model-b-workers`) while preserving one shared GPU frame per frame number and one Protobuf result per model.
- `src/pipeline/run_end_to_end.py` starts the Phase 1 PostgreSQL results processor, runs the extended DeepStream pipeline, waits for both model outputs in PostgreSQL, verifies all referenced JPEGs, and stops the processor gracefully.
- The original disposable Kafka verifier has been hardened to explicitly start reading at captured pre-publish end offsets, avoiding the `latest` startup race discovered in Phase 1.
- Every DeepStream run uses a fresh source-session UUID. **No deletion** of Kafka or PostgreSQL data is performed.

### Phase 4 — Performance / workload testing

- `src/monitoring/metrics.py`: thread-safe bounded latency histograms, p50/p95/p99 and counters.
- `src/monitoring/resource_sampler.py`: samples process CPU/RSS via `psutil` and GPU utilization/VRAM using `nvidia-smi` when available.
- `src/benchmark/kafka_load.py`: publishes valid binary Protobuf results for both models, same frame IDs, with configurable frame count and target FPS. Optional real database row-count audit.
- `src/benchmark/run_matrix.py`: compares multiple Kafka load rates, emits CSV plus per-run JSON.
- `src/benchmark/consumer_lag.py`: read-only per-partition Kafka consumer-group lag snapshot.
- `src/benchmark/gpu_scaling_matrix.py`: repeats the same MP4 with different Model A/B worker counts and emits a comparative GPU/latency CSV.

**Important distinction:** Passing unit tests proves the supporting logic, not that your local GPU, MP4 decoding, Kafka or PostgreSQL integration has been exercised with this patch. Run the real Docker test below before accepting the implementation. The three deferred Phase 1 outage/rebalance/backpressure tests remain deferred.

## 1. Install locally

From **your local checkout**, backed up if you have local edits:

```bash
cd /mnt/d/projects/deepstream-ai-monitoring
unzip -o /path/to/deepstream_phase234_patch.zip
```

This historical phase document describes the original incremental implementation. Your local Phase 1 v2 package must remain installed. Generated event JPEGs and reports will live under `artifacts/` on your bind-mounted host checkout.

Install required libraries inside the running `dev` container:

```bash
docker compose exec -T dev python3 -m pip install -r requirements.txt
```

The package deliberately does **not** change `requirements.txt` or `docker/Dockerfile`: reproducible rebuilds/Compose secret configuration are part of your postponed Phase 5. A container rebuild can remove manually installed libraries.

## 2. Run the CPU-only tests

```bash
docker compose exec -T dev python3 -m unittest -v \
  src.storage.event_frame_store_test \
  src.storage.verify_event_images_test \
  src.pipeline.run_event_pipeline_test \
  src.monitoring.metrics_test \
  src.benchmark.gpu_scaling_matrix_test \
  src.benchmark.kafka_load_test
```

**Local package validation:** 28/28 CPU tests passed here with Pillow, NumPy and psutil. This test environment has no running DeepStream/GPU/Kafka/PostgreSQL containers. The existing 49 Phase 1 tests were already run successfully in your Docker environment before this patch.

## 3. Run the complete DeepStream → Kafka → PostgreSQL test

Your PostgreSQL `DATABASE_PASSWORD` must still be set in the **host terminal**, as before; never paste it into this chat. Select a real MP4 visible under `/workspace` inside the `dev` container. For example, if the file is `videos/input.mp4` in your checkout, use `/workspace/videos/input.mp4`.

```bash
# From your project directory, with DATABASE_PASSWORD exported:
docker compose exec -e DATABASE_PASSWORD="$DATABASE_PASSWORD" -T dev \
  python3 -m src.pipeline.run_end_to_end \
  --source /workspace/videos/input.mp4 \
  --source-id camera_0 \
  --model-a-workers 1 --model-b-workers 1 \
  --event-root artifacts/event_frames \
  --event-min-confidence 0.0 \
  --event-every-n 30 \
  --db-timeout 180
```

`--event-min-confidence 0.0` is useful for **testing** with synthetic mock detections so event JPEGs are actually produced. The normal default is 0.65. This command processes the whole MP4, so duration depends on video length and GPU speed.

**Expected final line** (only after DB/image verification and graceful consumer shutdown):

```
PASS: DeepStream -> 2 GPU mock models -> Kafka -> PostgreSQL + JPEG storage
```

The command also prints `E2E session=<uuid>` and creates:

- `artifacts/event_frames/events/<camera-hash>/<session>/<frame>.jpg`
- `artifacts/benchmarks/phase234_<session>.json` (inference, Kafka enqueue, resource samples)
- `artifacts/benchmarks/end_to_end_<session>.json` (PostgreSQL record counts and JPEG verification)

To re-check image integrity later, replace `<session>` with the printed 32-character run UUID:

```bash
docker compose exec -e DATABASE_PASSWORD="$DATABASE_PASSWORD" -T dev \
  python3 -m src.storage.verify_event_images \
  --session-id <session> --event-root artifacts/event_frames
```

## 4. Benchmark Kafka → PostgreSQL throughput (separate terminals)

First, start the Phase 1 results processor in **Terminal A**:

```bash
docker compose exec -e DATABASE_PASSWORD="$DATABASE_PASSWORD" -T dev \
  python3 -m src.results.run_results_processor --config configs/config.yaml
```

Then, in **Terminal B** (same project directory):

```bash
# Send 500 frame identities, each with one Model A and one Model B message.
# This is 1,000 binary Protobuf messages at a TARGET 100 frames/sec.
docker compose exec -e DATABASE_PASSWORD="$DATABASE_PASSWORD" -T dev \
  python3 -m src.benchmark.kafka_load \
  --frames 500 --rate 100 --await-database

# Check actual consumer-group lag on all 6 model-topic partitions.
docker compose exec -T dev python3 -m src.benchmark.consumer_lag

# Compare three source rates; verify 1,000 DB rows for each run.
docker compose exec -e DATABASE_PASSWORD="$DATABASE_PASSWORD" -T dev \
  python3 -m src.benchmark.run_matrix \
  --rates 25,100,250 --frames 500
```

Stop Terminal A using `Ctrl+C` once complete. **The load generator intentionally publishes `is_event=False`: it must not invent nonexistent JPEG files.** All test rows remain in PostgreSQL unless you explicitly clean them up later. Every run has a unique source-session ID and deterministic per-model result IDs.

## 5. Compare actual DeepStream mock GPU worker counts

This repeatedly runs the full two-model pipeline and PostgreSQL check, so it is slower than the Kafka-only load test. Use a short MP4 at first:

```bash
docker compose exec -e DATABASE_PASSWORD="$DATABASE_PASSWORD" -T dev \
  python3 -m src.benchmark.gpu_scaling_matrix \
  --source /workspace/videos/input.mp4 \
  --source-id camera_0 \
  --profiles 1x1,2x1,1x2,2x2
```

The resulting `artifacts/benchmarks/gpu_scaling_<date>.csv` compares frame counts, elapsed time, per-model inference p95, Kafka enqueue p95, average observed GPU use, and event images.

Running more GPU workers means more CUDA streams and model weights in GPU memory. Increase counts slowly to avoid GPU OOM. The GPU benchmarks validate **mock-model orchestration only**, not real YOLO/ONNX/TensorRT model capacity.

## Operational notes and known limitations

1. **Phase 1 prerequisite:** Existing consumer modules and migration `003_deepstream_result_metadata.sql` must already be installed and applied (you previously verified this).
2. **Storage is local:** `artifacts/event_frames` must be on persistent, shared storage accessible to the frame publisher and any image reader. The existing dev Compose bind-mount `.:/workspace` makes it persistent on your WSL checkout, but this is not an object-storage service.
3. **Crash window:** The image is stored before its Kafka record is published. A process crash between those steps may leave an **unreferenced JPEG**. It will not create a dangling published reference. Orphan cleanup is deferred.
4. **Database schema:** The unique `(camera_id, frame_id)` key in `event_images` ensures both model events link to one image; PostgreSQL transaction/offset handling remains in Phase 1.
5. **No real model events:** Confidence/event logic currently operates on **synthetic mock detections**. Adapt policy when real models arrive.
6. **No production HA claim:** Live outage/rebalance/extended-load recovery tests from Phase 1 are still deferred, and this Phase 2–4 integration has not been tested against your GPU machine here.
7. **Phase 5 is not included:** Container dependency persistence, startup orchestration, `.env`/secret injection, final health checks and production packaging remain for the later phase.
8. **Generated files:** All of `artifacts/` is excluded by `.gitignore`.
9. **No GitHub changes were made.** You decide when to commit/push your locally tested code.

## File inventory

- `src/storage/{event_frame_store,event_policy,verify_event_images}.py` + tests
- `src/monitoring/{metrics,resource_sampler}.py` + tests
- `src/pipeline/{run_event_pipeline,run_end_to_end}.py` + tests
- `src/benchmark/{kafka_load,consumer_lag,run_matrix,gpu_scaling_matrix}.py` + tests
- `configs/config.yaml`
- `requirements.txt`
