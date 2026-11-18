"""Atomic JSON sidecars for stored event-frame JPEGs."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid
from typing import Any

from src.storage.event_frame_store import EventFrameStore


@dataclass(frozen=True, slots=True)
class EventMetadataFile:
    storage_key: str
    metadata_path: Path
    size_bytes: int
    sha256: str


@dataclass(slots=True)
class EventMetadataStore:
    """Write one integrity-checkable JSON sidecar beside each event JPEG."""

    frame_store: EventFrameStore
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def path_for_image(self, storage_key: str) -> Path:
        return self.frame_store.path_for_key(storage_key).with_suffix(".json")

    def write(self, *, storage_key: str, metadata: dict[str, Any]) -> EventMetadataFile:
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dictionary")
        image_metadata = metadata.get("image")
        if not isinstance(image_metadata, dict):
            raise ValueError("metadata.image must be an object")
        if image_metadata.get("storage_key") != storage_key:
            raise ValueError("metadata image.storage_key must match storage_key")

        target = self.path_for_image(storage_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        digest = hashlib.sha256(payload).hexdigest()

        with self._lock:
            if target.is_file():
                existing = target.read_bytes()
                if existing != payload:
                    raise ValueError(f"Conflicting event metadata already exists: {target}")
            else:
                temporary = target.with_name(
                    f".{target.name}.{uuid.uuid4().hex}.tmp"
                )
                try:
                    with temporary.open("xb") as handle:
                        handle.write(payload)
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)

        return EventMetadataFile(
            storage_key=storage_key,
            metadata_path=target,
            size_bytes=len(payload),
            sha256=digest,
        )

    def read(self, storage_key: str) -> dict[str, Any]:
        path = self.path_for_image(storage_key)
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            raise ValueError(f"Invalid event metadata document: {path}")
        return value

    def verify(self, storage_key: str) -> Path:
        path = self.path_for_image(storage_key)
        metadata = self.read(storage_key)
        image_metadata = metadata.get("image")
        if not isinstance(image_metadata, dict):
            raise ValueError(f"Event metadata has no image object: {path}")
        if image_metadata.get("storage_key") != storage_key:
            raise ValueError(f"Event metadata storage-key mismatch: {path}")
        return path
