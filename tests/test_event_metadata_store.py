import json

import numpy as np
import pytest

from src.storage.event_frame_store import EventFrameStore
from src.storage.event_metadata_store import EventMetadataStore


SESSION = "0123456789abcdef0123456789abcdef"


def test_event_metadata_is_written_next_to_jpeg_and_read_back(tmp_path):
    frame_store = EventFrameStore(tmp_path / "images", quality=88)
    metadata_store = EventMetadataStore(frame_store)
    image = frame_store.put_rgb(
        camera_id="camera-01",
        source_session_id=SESSION,
        frame_sequence=42,
        rgb_array=np.full((12, 16, 3), 127, dtype=np.uint8),
    )
    metadata = {
        "schema_version": "1.0",
        "camera_id": "camera-01",
        "event_id": "event-42",
        "image": {
            "storage_key": image.storage_key,
            "sha256": image.image_sha256,
            "size_bytes": image.image_size_bytes,
        },
        "detections": [{"class_name": "person", "confidence": 0.91}],
    }

    sidecar = metadata_store.write(
        storage_key=image.storage_key,
        metadata=metadata,
    )

    assert sidecar.metadata_path.is_file()
    assert sidecar.metadata_path.with_suffix(".jpg").is_file()
    assert json.loads(sidecar.metadata_path.read_text(encoding="utf-8")) == metadata
    assert metadata_store.verify(image.storage_key) == sidecar.metadata_path


def test_event_metadata_rejects_a_mismatched_storage_key(tmp_path):
    frame_store = EventFrameStore(tmp_path / "images")
    metadata_store = EventMetadataStore(frame_store)

    with pytest.raises(ValueError, match="must match"):
        metadata_store.write(
            storage_key="events/camera/session/000000000001.jpg",
            metadata={"image": {"storage_key": "different.jpg"}},
        )


def test_conflicting_sidecar_is_not_overwritten(tmp_path):
    frame_store = EventFrameStore(tmp_path / "images")
    metadata_store = EventMetadataStore(frame_store)
    image = frame_store.put_rgb(
        camera_id="camera-01",
        source_session_id=SESSION,
        frame_sequence=7,
        rgb_array=np.full((8, 8, 3), 100, dtype=np.uint8),
    )
    first = {
        "camera_id": "camera-01",
        "image": {"storage_key": image.storage_key},
        "event_id": "first",
    }
    second = {
        "camera_id": "camera-01",
        "image": {"storage_key": image.storage_key},
        "event_id": "second",
    }
    metadata_store.write(storage_key=image.storage_key, metadata=first)

    with pytest.raises(ValueError, match="Conflicting"):
        metadata_store.write(storage_key=image.storage_key, metadata=second)
