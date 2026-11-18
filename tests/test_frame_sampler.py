from src.pipeline.frame_sampler import FrameSampler


def test_30_fps_source_is_sampled_at_8_fps_by_pts():
    sampler = FrameSampler(target_fps=8.0, fallback_source_fps=30.0)
    sampled = []
    for frame_number in range(30):
        pts_ns = round(frame_number * 1_000_000_000 / 30)
        decision = sampler.decide(frame_number=frame_number, pts_ns=pts_ns)
        if decision.should_process:
            sampled.append(frame_number)

    assert sampled == [0, 4, 8, 12, 15, 19, 23, 27]


def test_repeated_pts_uses_frame_number_fallback():
    sampler = FrameSampler(target_fps=8.0, fallback_source_fps=30.0)
    decisions = [
        sampler.decide(frame_number=index, pts_ns=0)
        for index in range(5)
    ]

    assert [index for index, item in enumerate(decisions) if item.should_process] == [
        0, 4
    ]


def test_nonzero_initial_pts_is_used_as_time_origin():
    sampler = FrameSampler(target_fps=8.0, fallback_source_fps=30.0)
    sampled = []
    for frame_number in range(30):
        pts_ns = 900_000_000_000 + round(frame_number * 1_000_000_000 / 30)
        decision = sampler.decide(frame_number=frame_number, pts_ns=pts_ns)
        if decision.should_process:
            sampled.append(frame_number)

    assert sampled == [0, 4, 8, 12, 15, 19, 23, 27]


def test_sampler_rejects_invalid_rates():
    import pytest

    with pytest.raises(ValueError, match="target_fps"):
        FrameSampler(target_fps=0)
    with pytest.raises(ValueError, match="fallback_source_fps"):
        FrameSampler(target_fps=8, fallback_source_fps=-1)
