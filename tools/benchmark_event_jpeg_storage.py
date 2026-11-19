"""Compare event JPEG pair storage strategies on representative frames.

The measured interval starts when storage of one event begins and ends when
MinIO has acknowledged both the clean and boxed objects. Decoding the source
JPEG is outside that interval. Each candidate writes fresh object keys.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from src.storage.event_frame_store import EventFrameStore

BOXES = ((480.0, 220.0, 860.0, 980.0), (1040.0, 180.0, 1460.0, 1000.0))


def _load_env() -> None:
    for line in Path(".env").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    os.environ["OBJECT_STORE_ENDPOINT"] = os.environ.get(
        "OBJECT_STORE_ENDPOINT_HOST", "http://127.0.0.1:9000"
    )


def _frames(limit: int) -> list[np.ndarray]:
    paths = sorted(
        path for path in Path("artifacts/event_frames").rglob("*.jpg")
        if not path.name.endswith(".boxed.jpg")
    )
    if len(paths) < limit:
        raise SystemExit(f"Need {limit} clean event frames, found {len(paths)}")
    selected = paths[:limit]
    return [np.asarray(Image.open(path).convert("RGB")) for path in selected]


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    count = len(ordered)

    def percentile(rank: float) -> float:
        if count == 1:
            return ordered[0]
        position = (count - 1) * rank
        lower = int(position)
        weight = position - lower
        if lower >= count - 1:
            return ordered[-1]
        return ordered[lower] * (1.0 - weight) + ordered[lower + 1] * weight

    total = sum(ordered)
    return {
        "count": count,
        "mean_ms": statistics.fmean(ordered) * 1000.0,
        "p50_ms": percentile(0.50) * 1000.0,
        "p95_ms": percentile(0.95) * 1000.0,
        "events_per_second": count / total if total else 0.0,
    }


def _encode(image: Image.Image, quality: int) -> tuple[bytes, float]:
    started = time.perf_counter()
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=0)
    payload = buffer.getvalue()
    return payload, time.perf_counter() - started


def _digest(payload: bytes) -> tuple[str, float]:
    started = time.perf_counter()
    value = hashlib.sha256(payload).hexdigest()
    return value, time.perf_counter() - started


def _draw(image: Image.Image, boxes) -> float:
    started = time.perf_counter()
    draw = ImageDraw.Draw(image)
    for box in boxes:
        draw.rectangle(box, outline=(0, 255, 80), width=max(2, round(min(image.size) / 360)))
    return time.perf_counter() - started


class _Uploader:
    def __init__(self, store: EventFrameStore):
        self.store = store
        self.store_client = store.client

    def put(self, key: str, payload: bytes, digest: str, counter: str) -> float:
        started = time.perf_counter()
        try:
            self.store_client.put_object(
                Bucket=self.store.bucket, Key=key, Body=payload,
                ContentType="image/jpeg", IfNoneMatch="*",
            )
        except Exception as exc:
            response = getattr(exc, "response", None)
            code = response["ResponseMetadata"]["HTTPStatusCode"] if response else None
            if code != 412:
                raise
            self.store.verify(storage_key=key, image_size_bytes=len(payload), image_sha256=digest)
        return time.perf_counter() - started


def _key(store: EventFrameStore, session: str, sequence: int, boxed: bool) -> str:
    clean = f"{store.prefix}/{session}/{sequence:012d}.jpg"
    return store.boxed_key(clean) if boxed else clean


def baseline_put_rgb(store: EventFrameStore, frame: np.ndarray, session: str, sequence: int) -> dict:
    """Current production path: two independent put_rgb calls."""
    started = time.perf_counter()
    first = time.perf_counter()
    image = Image.fromarray(frame)
    copy_s = time.perf_counter() - first
    clean, clean_encode = _encode(image, store.quality)
    clean_hash, clean_hash_s = _digest(clean)
    clean_upload = _Uploader(store).put(_key(store, session, sequence, False), clean, clean_hash, "")
    second = time.perf_counter()
    boxed_image = Image.fromarray(frame)
    copy_s += time.perf_counter() - second
    draw_s = _draw(boxed_image, BOXES)
    boxed, boxed_encode = _encode(boxed_image, store.quality)
    boxed_hash, boxed_hash_s = _digest(boxed)
    boxed_upload = _Uploader(store).put(_key(store, session, sequence, True), boxed, boxed_hash, "boxed_")
    return {
        "pair_s": time.perf_counter() - started,
        "copy_s": copy_s,
        "clean_encode_s": clean_encode,
        "draw_s": draw_s,
        "boxed_encode_s": boxed_encode,
        "hash_s": clean_hash_s + boxed_hash_s,
        "clean_upload_s": clean_upload,
        "boxed_upload_s": boxed_upload,
    }


def single_copy_sequential(store: EventFrameStore, uploader: _Uploader, frame, session, sequence, quality) -> dict:
    started = time.perf_counter()
    image = Image.fromarray(frame)
    clean, clean_encode = _encode(image, quality)
    clean_hash, clean_hash_s = _digest(clean)
    draw_s = _draw(image, BOXES)
    boxed, boxed_encode = _encode(image, quality)
    boxed_hash, boxed_hash_s = _digest(boxed)
    clean_upload = uploader.put(_key(store, session, sequence, False), clean, clean_hash, "")
    boxed_upload = uploader.put(_key(store, session, sequence, True), boxed, boxed_hash, "boxed_")
    return {
        "pair_s": time.perf_counter() - started,
        "clean_encode_s": clean_encode,
        "draw_s": draw_s,
        "boxed_encode_s": boxed_encode,
        "hash_s": clean_hash_s + boxed_hash_s,
        "clean_upload_s": clean_upload,
        "boxed_upload_s": boxed_upload,
    }


def overlap_upload(store, uploader, frame, session, sequence, quality, pool: ThreadPoolExecutor) -> dict:
    """Encode the clean JPEG, upload it while the boxed JPEG is prepared, then upload that."""
    started = time.perf_counter()
    image = Image.fromarray(frame)
    clean, clean_encode = _encode(image, quality)
    clean_hash, clean_hash_s = _digest(clean)
    clean_key = _key(store, session, sequence, False)
    clean_future = pool.submit(uploader.put, clean_key, clean, clean_hash, "")
    draw_s = _draw(image, BOXES)
    boxed, boxed_encode = _encode(image, quality)
    boxed_hash, boxed_hash_s = _digest(boxed)
    boxed_upload = uploader.put(_key(store, session, sequence, True), boxed, boxed_hash, "boxed_")
    clean_upload = clean_future.result()
    return {
        "pair_s": time.perf_counter() - started,
        "clean_encode_s": clean_encode,
        "draw_s": draw_s,
        "boxed_encode_s": boxed_encode,
        "hash_s": clean_hash_s + boxed_hash_s,
        "clean_upload_s": clean_upload,
        "boxed_upload_s": boxed_upload,
    }


def concurrent_two_clients(store, clients, frame, session, sequence, quality, pool: ThreadPoolExecutor) -> dict:
    """One frame copy, then upload the two JPEGs at once through separate clients."""
    started = time.perf_counter()
    image = Image.fromarray(frame)
    clean, clean_encode = _encode(image, quality)
    clean_hash, clean_hash_s = _digest(clean)
    draw_s = _draw(image, BOXES)
    boxed, boxed_encode = _encode(image, quality)
    boxed_hash, boxed_hash_s = _digest(boxed)
    clean_uploader = _Uploader(store)
    clean_uploader.store_client = clients[0]
    boxed_uploader = _Uploader(store)
    boxed_uploader.store_client = clients[1]
    clean_future = pool.submit(
        clean_uploader.put, _key(store, session, sequence, False), clean, clean_hash, "")
    boxed_future = pool.submit(
        boxed_uploader.put, _key(store, session, sequence, True), boxed, boxed_hash, "boxed_")
    clean_upload = clean_future.result()
    boxed_upload = boxed_future.result()
    return {
        "pair_s": time.perf_counter() - started,
        "clean_encode_s": clean_encode,
        "draw_s": draw_s,
        "boxed_encode_s": boxed_encode,
        "hash_s": clean_hash_s + boxed_hash_s,
        "clean_upload_s": clean_upload,
        "boxed_upload_s": boxed_upload,
    }


def concurrent_uploads(store, uploader, frame, session, sequence, quality, pool: ThreadPoolExecutor) -> dict:
    started = time.perf_counter()
    image = Image.fromarray(frame)
    clean, clean_encode = _encode(image, quality)
    clean_hash, clean_hash_s = _digest(clean)
    draw_s = _draw(image, BOXES)
    boxed, boxed_encode = _encode(image, quality)
    boxed_hash, boxed_hash_s = _digest(boxed)
    clean_future = pool.submit(uploader.put, _key(store, session, sequence, False), clean, clean_hash, "")
    boxed_future = pool.submit(uploader.put, _key(store, session, sequence, True), boxed, boxed_hash, "boxed_")
    clean_upload = clean_future.result()
    boxed_upload = boxed_future.result()
    return {
        "pair_s": time.perf_counter() - started,
        "clean_encode_s": clean_encode,
        "draw_s": draw_s,
        "boxed_encode_s": boxed_encode,
        "hash_s": clean_hash_s + boxed_hash_s,
        "clean_upload_s": clean_upload,
        "boxed_upload_s": boxed_upload,
    }


def _print_report(name: str, rows: list[dict]) -> None:
    summary = _summary([row["pair_s"] for row in rows])
    print(
        f"{name:28} mean={summary['mean_ms']:.2f} ms  "
        f"p50={summary['p50_ms']:.2f}  p95={summary['p95_ms']:.2f}  "
        f"events/s={summary['events_per_second']:.2f}"
    )
    numeric = [key for key in rows[0] if key.endswith("_s") and key != "pair_s"]
    if numeric:
        parts = []
        for key in numeric:
            parts.append(f"{key[:-2]}={statistics.fmean(row[key] for row in rows) * 1000.0:.2f} ms")
        print("  " + "  ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    _load_env()
    frames = _frames(args.frames + args.warmup)
    store = EventFrameStore("s3://camera-events/events/camera-01", quality=88)
    uploader = _Uploader(store)
    clients = (store.client, EventFrameStore("s3://camera-events/events/camera-01", quality=88).client)
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="jpeg-bench")
    stamp = int(time.time())
    strategies = (
        ("baseline_two_puts", lambda index, frame: baseline_put_rgb(
            store, frame, f"bench-baseline-{stamp}", index)),
        ("single_copy_sequential", lambda index, frame: single_copy_sequential(
            store, uploader, frame, f"bench-sequential-{stamp}", index, store.quality)),
        ("overlap_clean_upload", lambda index, frame: overlap_upload(
            store, uploader, frame, f"bench-overlap-{stamp}", index, store.quality, pool)),
        ("concurrent_one_client", lambda index, frame: concurrent_uploads(
            store, uploader, frame, f"bench-concurrent-{stamp}", index, store.quality, pool)),
        ("concurrent_two_clients", lambda index, frame: concurrent_two_clients(
            store, clients, frame, f"bench-two-client-{stamp}", index, store.quality, pool)),
    )
    try:
        for index, frame in enumerate(frames[:args.warmup]):
            for run in strategies:
                run[1](20_000 + index, frame)
        measured = {name: [] for name, _ in strategies}
        for index, frame in enumerate(frames[args.warmup:]):
            order = list(strategies)
            if index % 2:
                order.reverse()
            for name, run in order:
                measured[name].append(run(index, frame))
        for name, _ in strategies:
            _print_report(name, measured[name])
    finally:
        pool.shutdown(wait=True, cancel_futures=False)


if __name__ == "__main__":
    main()
