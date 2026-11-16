# Phase 1 — Kafka Reliability Patch

This is an **overlay patch** for the existing `deepstream-ai-monitoring`
repository on your `DeepStream_Pipeline` branch, building on the previously
installed Kafka asynchronous batch module. It is **not** a standalone repo and
does **not** replace your DeepStream producer, Protobuf-generated module,
PostgreSQL migration, or `config/kafka.json`.

No GitHub changes were made. No passwords are included.

## Changed / new files

| File | Purpose |
| --- | --- |
| `src/kafka/consumer.py` | Kafka assignment/revocation callbacks, epoch-based stale-commit fencing, paused polling for group heartbeat/callbacks, new-partition pause coverage, explicit commit ownership checks |
| `src/kafka/records.py` | Preserve the assignment epoch alongside each consumed record |
| `src/kafka/async_batch_writer.py` | Prompt shutdown flushing even with a long timer; do not flush partial batches prematurely on periodic wakeups |
| `src/results/results_processing_service.py` | Periodic Kafka polling under backpressure and while the writer drains, assignment fencing before enqueue/commit, reliability metrics |
| `src/kafka/consumer_test.py` | Assignment revocation/loss, stale and unassigned commit tests, pause and error behavior |
| `src/kafka/async_batch_writer_test.py` | Long flush interval, shutdown latency, and DLT failure tests |
| `src/kafka/dead_letter_publisher_test.py` | DLT source metadata/payload preservation and delivery failure checks |
| `src/results/results_processing_service_test.py` | Consumer→async worker→ack tests, simulated DB failure, DLT failure, commit failure, rebalance, backpressure |
| `src/results/phase1_dead_letter_smoke.py` | Real Kafka DLT integration test (one intentionally invalid source message) |

## Install

Back up any locally changed versions of those files first, then extract the
ZIP **at the repository root**, preserving paths and overwriting the indicated
files. For example from `/mnt/d/projects/deepstream-ai-monitoring`:

```bash
unzip -l /path/to/kafka_phase1_reliability_patch.zip
unzip -o /path/to/kafka_phase1_reliability_patch.zip
```

You do **not** need to apply another SQL migration for Phase 1, change existing
Kafka topics, rebuild the Docker container, or disclose the database password.
The new Python files are picked up immediately if the Compose service already
bind-mounts your repo to `/workspace`, as it did in earlier tests. Avoid
extracting over active consumers; stop them first.

## Unit and simulated-failure tests

From the repository root:

```bash
docker compose exec -T dev python3 -m unittest -v \
  src.kafka.protobuf_codec_test \
  src.kafka.message_validator_test \
  src.kafka.async_batch_writer_test \
  src.kafka.consumer_test \
  src.kafka.dead_letter_publisher_test \
  src.results.results_processing_service_test
```

Expected: **44 tests**. Tests intentionally simulate DB and DLT outages; some
log `Fatal batch processing failure` yet should still report `OK`.

## Real Kafka dead-letter smoke test

With other publishers stopped and Kafka healthy, run from your repository root:

```bash
docker compose exec -e DATABASE_PASSWORD="$DATABASE_PASSWORD" -T dev \
  python3 -m src.results.phase1_dead_letter_smoke \
  --config config/kafka.json
```

The test:

1. Starts a fresh Kafka consumer group at latest offsets, with a
   notification when the result-topic partition assignment is ready.
2. Captures current DLT high-water marks so prior test messages are ignored.
3. Produces **one intentionally malformed Protobuf** message to the model-A
   topic (the existing topic is not deleted or recreated).
4. Uses the real results consumer, validator, asynchronous writer and DLT
   publisher. Valid DB writes are deliberately rejected during this test.
5. Verifies the DLT event ID, original payload bytes, and committed source
   offset = original offset + 1.

**The test leaves its malformed message and DLT envelope in Kafka according to
normal retention.** It does not create PostgreSQL rows. Do not run with active
DeepStream producers: the fresh test group could consume another producer's
valid message before its intentional invalid one.

Expected final line:

```
PASS: real Kafka dead-letter + offset integration smoke test
```

## What is and is not guaranteed

- **Normal path:** Database transaction or DLT delivery must complete before
  the consumer thread commits offsets. This is **at-least-once processing**,
  not an atomic cross-system transaction.
- **Database or DLT outage:** After bounded retries, the writer fails and the
  processor exits nonzero without acknowledging the failed batch. Restart it
  after restoring the dependency to replay. An external service restart policy
  is required for unattended recovery; this patch does not supply one.
- **Commit failure:** DB may already be committed; Kafka can replay the same
  message. PostgreSQL result idempotency must be preserved. DLT delivery can
  be duplicated on replay; downstream consumers should deduplicate on the
  stable `event_id`.
- **Rebalance / lost assignment:** Previously fetched records carry a
  conservative assignment epoch. Revocations/loss invalidate older batches;
  stale completions are not committed. The service fails closed and relies on
  Kafka replay and DB idempotency. It does not attempt seamless multi-instance
  rebalancing with in-flight work.
- **Backpressure:** Pauses assigned partitions and continues polling for group
  events rather than sleeping. An unexpected data message while paused
  causes a fail-safe stop without committing that message.
- **Shutdown:** Queued records flush without waiting for the full flush timer.
  This is cooperative shutdown; a hard kill can still interrupt a transaction,
  in which case Kafka redelivery is expected.

## Still to validate in your WSL2 Docker environment

These are not claimed as live-tested by the ZIP author:

- Real Kafka DLT delivery (run the smoke command above).
- Real PostgreSQL outage/restart and consumer-group rebalance under a genuine
  multi-consumer workload.
- Sustained load, shutdown signals under high load, and latency/lag limits.

There is no need to delete the two rows created by your previous successful
Kafka→PostgreSQL test. This patch does not touch existing DB data.

Phases 2–5 (event images, full DeepStream integration, load testing, reproducible
configuration) are **not included**. They can be implemented separately and
combined into a final ZIP after local verification.
