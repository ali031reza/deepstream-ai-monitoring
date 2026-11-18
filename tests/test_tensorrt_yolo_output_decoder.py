import numpy as np

from src.models.tensorrt_yolo11n_engine import TensorRTYOLO11nEngine


def _decode(output):
    return TensorRTYOLO11nEngine.decode_yolo_output(
        output,
        frame_shape=(640, 640),
        ratio=1.0,
        pad_left=0,
        pad_top=0,
        confidence=0.35,
        iou=0.45,
        classes=(0,),
        max_detections=100,
        class_names={0: "person"},
    )


def test_yolo_channels_first_output_filters_person_and_suppresses_duplicates():
    output = np.zeros((1, 84, 8400), dtype=np.float32)

    output[0, 0:4, 0] = [320.0, 320.0, 120.0, 160.0]
    output[0, 4, 0] = 0.95
    output[0, 0:4, 1] = [320.0, 320.0, 120.0, 160.0]
    output[0, 4, 1] = 0.80

    output[0, 0:4, 2] = [100.0, 100.0, 40.0, 50.0]
    output[0, 5, 2] = 0.99

    detections = _decode(output)

    assert len(detections) == 1
    assert detections[0].class_id == 0
    assert detections[0].class_name == "person"
    assert detections[0].confidence == 0.95
    assert detections[0].bbox_normalized == (
        260.0 / 640.0,
        240.0 / 640.0,
        380.0 / 640.0,
        400.0 / 640.0,
    )


def test_yolo_candidates_first_output_is_also_supported():
    output = np.zeros((1, 8400, 84), dtype=np.float32)
    output[0, 0, 0:4] = [320.0, 320.0, 120.0, 160.0]
    output[0, 0, 4] = 0.90

    detections = _decode(output)

    assert len(detections) == 1
    assert detections[0].class_name == "person"


def test_yolo_output_decoder_returns_empty_for_below_threshold_scores():
    output = np.zeros((1, 84, 8400), dtype=np.float32)
    output[0, 0:4, 0] = [320.0, 320.0, 100.0, 100.0]
    output[0, 4, 0] = 0.20

    assert _decode(output) == []
