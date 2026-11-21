"""Store event JPEGs as immutable S3 objects in the camera's configured prefix.

A qualifying event stores one clean JPEG and one boxed JPEG. Both objects are
encoded from a single frame copy, then uploaded together. The call returns
only after MinIO has acknowledged both objects, so a Kafka result is never
published for a partial pair. A retry of an existing object keeps the first
durable JPEG and fills in the one that failed.
"""
from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import PurePosixPath
import threading
import time
from urllib.parse import urlsplit

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class EventImage:
    storage_key: str
    image_format: str
    image_size_bytes: int
    image_sha256: str


def _object_store_client():
    from configs.settings import load_settings

    return boto3.client(
        "s3",
        config=Config(s3={"addressing_style": "path"}),
        **load_settings().object_store.client_kwargs(),
    )


class EventFrameStore:
    def __init__(self, root: str, *, quality: int = 88, client=None):
        uri = urlsplit(root)
        self.prefix = uri.path.strip("/")
        if uri.scheme != "s3" or not uri.netloc or not self.prefix or uri.query or uri.fragment:
            raise ValueError(f"Expected s3://bucket/prefix, got {root!r}")
        self.bucket = uri.netloc
        self.root = root.rstrip("/")
        self.quality = quality
        self.client = client or _object_store_client()
        if client is None:
            self._ensure_bucket()
        self._stats_lock = threading.Lock()
        self._stats = dict(saved=0, reused=0, boxed_saved=0, boxed_reused=0, bytes_written=0)
        self._pair_samples: list[dict[str, float]] = []
        self._upload_pool: ThreadPoolExecutor | None = None

    def _ensure_bucket(self) -> None:
        try:
            self.client.head_bucket(Bucket=self.bucket)
        except ClientError as exc:
            if exc.response["ResponseMetadata"]["HTTPStatusCode"] != 404:
                raise
            try:
                self.client.create_bucket(Bucket=self.bucket)
            except ClientError as conflict:
                if conflict.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
                    raise

    def _executor(self) -> ThreadPoolExecutor:
        if self._upload_pool is None:
            self._upload_pool = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="event-jpeg-upload"
            )
            atexit.register(self._upload_pool.shutdown, wait=True)
        return self._upload_pool

    @staticmethod
    def boxed_key(clean_key):
        """Legacy boxed object name. Existing rows keep this key."""
        return str(PurePosixPath(clean_key).with_suffix(".boxed.jpg"))

    @staticmethod
    def boxed_key_for(clean_key: str, model_id: str) -> str:
        """Boxed object for one model, so another model can keep its own annotations."""
        model = model_id.strip()
        if not model or "/" in model or "\\" in model or model in {".", ".."}:
            raise ValueError("model_id must be a single path segment")
        stem = str(PurePosixPath(clean_key).with_suffix(""))
        return f"{stem}.{model}.boxed.jpg"

    def _encode(self, image: Image.Image) -> bytes:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality, subsampling=0)
        return buffer.getvalue()

    @staticmethod
    def _draw_boxes(image: Image.Image, boxes) -> None:
        draw = ImageDraw.Draw(image)
        for box in boxes:
            draw.rectangle(box, outline=(0, 255, 80), width=max(2, round(min(image.size) / 360)))

    def _put_object(self, key: str, payload: bytes, digest: str, counter: str, submitted_at: float) -> dict:
        queue_wait_s = time.perf_counter() - submitted_at
        upload_started = time.perf_counter()
        try:
            self.client.put_object(
                Bucket=self.bucket, Key=key, Body=payload,
                ContentType="image/jpeg", IfNoneMatch="*",
            )
        except ClientError as exc:
            if exc.response["ResponseMetadata"]["HTTPStatusCode"] != 412:
                raise
            self.verify(storage_key=key, image_size_bytes=len(payload), image_sha256=digest)
            with self._stats_lock:
                self._stats[counter + "reused"] += 1
        else:
            with self._stats_lock:
                self._stats[counter + "saved"] += 1
                self._stats["bytes_written"] += len(payload)
        return {
            "queue_wait_s": queue_wait_s,
            "upload_s": time.perf_counter() - upload_started,
        }

    def put_rgb(self, *, source_session_id, frame_sequence, rgb_array, boxes=None):
        """Store one JPEG. Event pairs go through put_pair so both uploads overlap."""
        key = f"{self.prefix}/{source_session_id}/{frame_sequence:012d}.jpg"
        image = Image.fromarray(rgb_array)
        counter = ""
        if boxes is not None:
            key = self.boxed_key(key)
            counter = "boxed_"
            self._draw_boxes(image, boxes)
        payload = self._encode(image)
        digest = hashlib.sha256(payload).hexdigest()
        self._put_object(key, payload, digest, counter, time.perf_counter())
        return EventImage(key, "jpeg", len(payload), digest)

    def put_pair(self, *, source_session_id, frame_sequence, rgb_array, boxes, model_id=None):
        """Encode the clean and boxed JPEGs, upload them together, and wait for both acks."""
        started = time.perf_counter()
        clean_key = f"{self.prefix}/{source_session_id}/{frame_sequence:012d}.jpg"
        boxed_key = (
            self.boxed_key_for(clean_key, model_id) if model_id else self.boxed_key(clean_key)
        )
        image = Image.fromarray(rgb_array)

        encode_started = time.perf_counter()
        clean_payload = self._encode(image)
        clean_encode_s = time.perf_counter() - encode_started
        hash_started = time.perf_counter()
        clean_digest = hashlib.sha256(clean_payload).hexdigest()
        clean_hash_s = time.perf_counter() - hash_started

        draw_started = time.perf_counter()
        self._draw_boxes(image, boxes)
        draw_s = time.perf_counter() - draw_started
        encode_started = time.perf_counter()
        boxed_payload = self._encode(image)
        boxed_encode_s = time.perf_counter() - encode_started
        hash_started = time.perf_counter()
        boxed_digest = hashlib.sha256(boxed_payload).hexdigest()
        boxed_hash_s = time.perf_counter() - hash_started

        pool = self._executor()
        drain_started = time.perf_counter()
        clean_future = pool.submit(
            self._put_object, clean_key, clean_payload, clean_digest, "", time.perf_counter()
        )
        boxed_future = pool.submit(
            self._put_object, boxed_key, boxed_payload, boxed_digest, "boxed_", time.perf_counter()
        )
        clean_meta, boxed_meta = self._collect_uploads(clean_future, boxed_future)
        drain_s = time.perf_counter() - drain_started
        pair_s = time.perf_counter() - started
        self._pair_samples.append({
            "pair_s": pair_s,
            "clean_encode_s": clean_encode_s,
            "draw_s": draw_s,
            "boxed_encode_s": boxed_encode_s,
            "hash_s": clean_hash_s + boxed_hash_s,
            "clean_upload_s": clean_meta["upload_s"],
            "boxed_upload_s": boxed_meta["upload_s"],
            "queue_wait_s": clean_meta["queue_wait_s"] + boxed_meta["queue_wait_s"],
            "drain_s": drain_s,
        })
        return (
            EventImage(clean_key, "jpeg", len(clean_payload), clean_digest),
            EventImage(boxed_key, "jpeg", len(boxed_payload), boxed_digest),
        )

    @staticmethod
    def _collect_uploads(clean_future, boxed_future):
        errors = []
        metas = []
        for future in (clean_future, boxed_future):
            try:
                metas.append(future.result())
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
        return metas

    def verify(self, *, storage_key, image_size_bytes, image_sha256):
        if not storage_key.startswith(self.prefix + "/"):
            raise ValueError("Object is outside the configured camera prefix")
        response = self.client.get_object(Bucket=self.bucket, Key=storage_key)
        with response["Body"] as body:
            digest = hashlib.sha256(body.read()).hexdigest()
        if response["ContentLength"] != image_size_bytes or digest != image_sha256:
            raise ValueError(f"Stored JPEG integrity mismatch: {storage_key}")

    def stats(self):
        with self._stats_lock:
            report = self._stats.copy()
        if self._pair_samples:
            report["pair_latency"] = _latency_summary([sample["pair_s"] for sample in self._pair_samples])
            report["stage_mean_ms"] = {
                name.removesuffix("_s"): _mean(sample[name] for sample in self._pair_samples) * 1000.0
                for name in self._pair_samples[0]
                if name != "pair_s"
            }
        return report


def _mean(values) -> float:
    total = 0.0
    count = 0
    for value in values:
        total += value
        count += 1
    return total / count if count else 0.0


def _percentile(ordered: list[float], rank: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * rank
    lower = int(position)
    weight = position - lower
    if lower >= len(ordered) - 1:
        return ordered[-1]
    return ordered[lower] * (1.0 - weight) + ordered[lower + 1] * weight


def _latency_summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    total = sum(ordered)
    return {
        "count": len(ordered),
        "mean_ms": _mean(ordered) * 1000.0,
        "p50_ms": _percentile(ordered, 0.50) * 1000.0,
        "p95_ms": _percentile(ordered, 0.95) * 1000.0,
        "events_per_second": len(ordered) / total if total else 0.0,
    }
