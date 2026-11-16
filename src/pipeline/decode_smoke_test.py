
import argparse
from pathlib import Path
from pprint import pprint

from pyservicemaker import Flow, Pipeline, Probe, RenderMode

from src.pipeline.frame_metadata_probe import FrameMetadataProbe


# =========================================================
# Pipeline configuration
# =========================================================

BATCH_SIZE = 1

# Preserve the original camera resolution.
BATCH_WIDTH = 1280
BATCH_HEIGHT = 720


def main():

    # =====================================================
    # 1. Parse command-line arguments
    # =====================================================

    parser = argparse.ArgumentParser(
        description="DeepStream single-camera decoding and metadata test"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to the video file inside the container",
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="NVIDIA GPU ID",
    )

    args = parser.parse_args()

    # =====================================================
    # 2. Validate video source
    # =====================================================

    video_path = Path(args.source)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    print("=" * 60)
    print("DEEPSTREAM VIDEO DECODING AND METADATA TEST")
    print("=" * 60)

    print(f"Video: {video_path}")
    print(f"GPU ID: {args.gpu_id}")

    print("\nPipeline configuration:")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Batch resolution: {BATCH_WIDTH}x{BATCH_HEIGHT}")

    # =====================================================
    # 3. Initialize metadata handler
    # =====================================================

    metadata_handler = FrameMetadataProbe()

    # =====================================================
    # 4. Create DeepStream probe
    # =====================================================

    metadata_probe = Probe(
        "frame_metadata_probe",
        metadata_handler,
    )

    # =====================================================
    # 5. Initialize DeepStream pipeline
    # =====================================================

    pipeline = Pipeline("single-camera-decode")

    # =====================================================
    # 6. Construct processing flow
    # =====================================================

    flow = (
        Flow(pipeline)
        .capture(
            [str(video_path)],
            gpu_id=args.gpu_id,
        )
        .batch(
            batch_size=BATCH_SIZE,
            width=BATCH_WIDTH,
            height=BATCH_HEIGHT,
        )
        .attach(metadata_probe)
        .render(mode=RenderMode.DISCARD)
    )

    # =====================================================
    # 7. Execute pipeline
    # =====================================================

    print("\nStarting DeepStream pipeline...\n")

    flow()

    # =====================================================
    # 8. Retrieve frame statistics
    # =====================================================

    summary = metadata_handler.get_summary()

    print("\n" + "=" * 60)
    print("FRAME PROCESSING STATISTICS")
    print("=" * 60)

    print(f"Total frames: {summary['total_frames']}")
    print(f"Elapsed seconds: {summary['elapsed_seconds']:.3f}")
    print(f"Observed FPS: {summary['observed_fps']:.2f}")

    # =====================================================
    # 9. Display sample frame metadata
    # =====================================================

    print("\n" + "=" * 60)
    print("SAMPLE FRAME METADATA")
    print("=" * 60)

    if summary["sample_frames"]:

        for index, frame in enumerate(
            summary["sample_frames"], start=1
        ):
            print(f"\nFrame sample {index}:")
            pprint(frame, sort_dicts=False)

    else:
        print("No frame metadata was collected.")

    # =====================================================
    # 10. Completion
    # =====================================================

    print("\n" + "=" * 60)
    print("VIDEO PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
