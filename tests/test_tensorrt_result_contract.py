from src.kafka.message_validator import validate_inference_result
from src.results.protobuf_result_adapter import build_inference_result


SESSION = "0123456789abcdef0123456789abcdef"


def test_yolo_event_result_matches_existing_protobuf_contract():
    result = build_inference_result(
        {
            "model_id": "yolo11n-person",
            "frame_id": 123,
            "gpu_id": 0,
            "input_frame_shape": [720, 1280, 3],
            "model_input_size": 640,
            "status": "processed",
            "is_event": True,
            "event_type": "person_detected",
            "storage_key": (
                "events/0123456789abcdef/"
                f"{SESSION}/000000000123.jpg"
            ),
            "image_format": "jpeg",
            "image_size_bytes": 2048,
            "image_sha256": "a" * 64,
            "detections": [
                {
                    "class_id": 0,
                    "class_name": "person",
                    "confidence": 0.91,
                    "bbox_normalized": [0.1, 0.2, 0.5, 0.9],
                    "synthetic": False,
                }
            ],
            "timings_ms": {
                "queue_wait_ms": 1.25,
                "tensorrt_inference_ms": 4.5,
            },
        },
        camera_id="camera-01",
        source_session_id=SESSION,
        frame_sequence=123,
        captured_at_ms=1_800_000_000_000,
        processed_at_ms=1_800_000_000_010,
        worker_id="tensorrt-yolo11n-worker-01",
    )

    validate_inference_result(
        result,
        allowed_model_ids={"yolo11n-person"},
        require_schema_version=True,
    )

    assert result.camera_id == "camera-01"
    assert result.model_id == "yolo11n-person"
    assert result.is_event is True
    assert result.event_type == "person_detected"
    assert result.detections[0].class_name == "person"
    assert result.detections[0].synthetic is False
    assert result.captured_at_ms == 1_800_000_000_000
    assert result.processed_at_ms == 1_800_000_000_010


def test_yolo_non_event_result_has_no_image_reference():
    result = build_inference_result(
        {
            "model_id": "yolo11n-person",
            "frame_id": 124,
            "gpu_id": 0,
            "input_frame_shape": [720, 1280, 3],
            "model_input_size": 640,
            "status": "processed",
            "is_event": False,
            "detections": [],
            "timings_ms": {"tensorrt_inference_ms": 4.5},
        },
        camera_id="camera-01",
        source_session_id=SESSION,
        frame_sequence=124,
        captured_at_ms=1_800_000_000_000,
        processed_at_ms=1_800_000_000_010,
        worker_id="tensorrt-yolo11n-worker-01",
    )

    validate_inference_result(
        result,
        allowed_model_ids={"yolo11n-person"},
        require_schema_version=True,
    )
    assert result.is_event is False
    assert not result.storage_key
    assert len(result.detections) == 0
