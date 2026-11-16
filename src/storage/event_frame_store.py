"""Phase 2: atomic, idempotent JPEG event-frame storage.

The image is stored BEFORE its Kafka result is published. Only metadata
references, not image bytes, travel through Kafka. The root directory must
be mounted persistently for production-like testing.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import os
from pathlib import Path, PurePosixPath
import re
import threading
import uuid


_SESSION = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class EventImage:
    storage_key: str
    image_format: str
    image_size_bytes: int
    image_sha256: str


class EventFrameStore:
    """Filesystem storage with one stable image per camera/session/frame.

    A mutex serializes writes by the two model workers (correctness first).
    Reusing a key with *different* JPEG bytes is a fatal conflict.
    """

    def __init__(self, root: str | Path, *, quality: int = 88):
        self.root = Path(root).expanduser().resolve()
        if not 1 <= quality <= 95:
            raise ValueError("JPEG quality must be in [1, 95]")
        self.quality = quality
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._stats = {"saved": 0, "reused": 0, "bytes_written": 0}

    @staticmethod
    def key(camera_id: str, source_session_id: str, frame_sequence: int) -> str:
        if not camera_id or len(camera_id) > 50:
            raise ValueError("camera_id must contain 1-50 characters")
        if not _SESSION.fullmatch(source_session_id):
            raise ValueError("source_session_id must be a 32-character hex UUID")
        if not isinstance(frame_sequence, int) or frame_sequence < 0:
            raise ValueError("frame_sequence must be nonnegative")
        camera_hash = hashlib.sha256(camera_id.encode("utf-8")).hexdigest()[:16]
        return f"events/{camera_hash}/{source_session_id}/{frame_sequence:012d}.jpg"

    def path_for_key(self, key: str) -> Path:
        path = PurePosixPath(key)
        if (
            not key or path.is_absolute() or "\\" in key or
            any(piece in ("", ".", "..") for piece in key.split("/")) or
            path.suffix.lower() not in (".jpg", ".jpeg")
        ):
            raise ValueError("Unsafe JPEG storage key")
        resolved = (self.root / Path(*path.parts)).resolve()
        if not resolved.is_relative_to(self.root):
            raise ValueError("JPEG storage key escapes configured root")
        return resolved

    def put_rgb(self, *, camera_id: str, source_session_id: str,
                frame_sequence: int, rgb_array) -> EventImage:
        """Accept HxWx3 RGB uint8 numpy/CuPy-compatible image.

        CuPy input is explicitly copied to CPU exactly when an event qualifies.
        The caller must retain the GPU frame until this method returns.
        """
        key = self.key(camera_id, source_session_id, frame_sequence)
        target = self.path_for_key(key)
        with self._lock:
            # Lazy imports keep storage metadata/verifier tools GPU-independent.
            import numpy as np
            from PIL import Image
            if hasattr(rgb_array, "get") and not isinstance(rgb_array, np.ndarray):
                rgb_array = rgb_array.get()
            array = np.asarray(rgb_array)
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                raise ValueError("Event frame must have shape (H,W,3) and uint8 dtype")
            if array.shape[0] < 1 or array.shape[1] < 1:
                raise ValueError("Event image is empty")
            buf = BytesIO()
            Image.fromarray(array, mode="RGB").save(
                buf, format="JPEG", quality=self.quality, subsampling=0,
            )
            payload = buf.getvalue()
            digest = hashlib.sha256(payload).hexdigest()
            if target.is_file():
                existing = target.read_bytes()
                if hashlib.sha256(existing).hexdigest() != digest:
                    raise ValueError(f"Image collision for {key}")
                self._stats["reused"] += 1
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Exclusive unique temporary file, then atomic rename.
                temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
                try:
                    with temporary.open("xb") as handle:
                        handle.write(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
                self._stats["saved"] += 1
                self._stats["bytes_written"] += len(payload)
        return EventImage(key, "jpeg", len(payload), digest)

    def verify(self, *, storage_key: str, image_size_bytes: int,
               image_sha256: str) -> Path:
        target = self.path_for_key(storage_key)
        digest = hashlib.sha256()
        size = 0
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        if size != image_size_bytes or digest.hexdigest() != image_sha256.lower():
            raise ValueError(f"Stored event image integrity mismatch: {storage_key}")
        return target

    def stats(self):
        with self._lock:
            return dict(self._stats)
