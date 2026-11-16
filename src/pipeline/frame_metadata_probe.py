import time

from pyservicemaker import BatchMetadataOperator


class FrameMetadataProbe(BatchMetadataOperator):
    """
    Lightweight observer for decoded video frame metadata.

    Does not copy, modify, or store video image data.
    """

    SAMPLE_LIMIT = 5

    def __init__(self):
        super().__init__()

        self.total_frames = 0

        self.first_observed_ns = None
        self.last_observed_ns = None

        # Keep only a small sample for inspection.
        self.sample_frames = []

    def handle_metadata(self, batch_meta):

        # One timestamp per incoming batch.
        observed_at_unix_ns = time.time_ns()
        observed_at_monotonic_ns = time.monotonic_ns()

        for frame_meta in batch_meta.frame_items:

            self.total_frames += 1

            if self.first_observed_ns is None:
                self.first_observed_ns = observed_at_monotonic_ns

            self.last_observed_ns = observed_at_monotonic_ns

            # Store only the first five metadata records.
            if len(self.sample_frames) < self.SAMPLE_LIMIT:

                frame_record = {
                    "frame_number": int(frame_meta.frame_number),
                    "source_id": int(frame_meta.source_id),
                    "pad_index": int(frame_meta.pad_index),
                    "batch_id": int(frame_meta.batch_id),

                    "buffer_pts": int(frame_meta.buffer_pts),

                    "observed_at_unix_ns": observed_at_unix_ns,
                    "observed_at_monotonic_ns": observed_at_monotonic_ns,

                    "source_width": int(frame_meta.source_width),
                    "source_height": int(frame_meta.source_height),

                    "pipeline_width": int(frame_meta.pipeline_width),
                    "pipeline_height": int(frame_meta.pipeline_height),
                }

                self.sample_frames.append(frame_record)

    def get_summary(self):

        elapsed_ns = 0

        if self.first_observed_ns is not None:
            elapsed_ns = (
                self.last_observed_ns - self.first_observed_ns
            )

        elapsed_seconds = elapsed_ns / 1_000_000_000

        observed_fps = 0.0

        if self.total_frames > 1 and elapsed_seconds > 0:
            observed_fps = (
                (self.total_frames - 1) / elapsed_seconds
            )

        return {
            "total_frames": self.total_frames,
            "elapsed_seconds": elapsed_seconds,
            "observed_fps": observed_fps,
            "sample_frames": list(self.sample_frames),
        }
