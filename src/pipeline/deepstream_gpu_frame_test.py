"""
DeepStream GPU Frame Extraction Test.

Objectives:
    1. Decode the original video using DeepStream.
    2. Retrieve actual decoded buffers using BufferRetriever.
    3. Extract frame tensors using buffer.extract().
    4. Import tensors into CuPy through DLPack.
    5. Verify GPU device, dimensions, dtype and memory.
    6. Execute a small GPU operation on the extracted frame.

Important:
    All GPU operations are completed inside consume().
    The DeepStream buffer must not be used after consume returns.
"""

import argparse
import statistics
import sys
import time

from pathlib import Path

from pyservicemaker import (
    Flow,
    Pipeline,
    BufferRetriever,
)


BATCH_SIZE = 1
BATCH_WIDTH = 1280
BATCH_HEIGHT = 720


class GPUFrameInspectionRetriever(BufferRetriever):

    def __init__(self, cp, gpu_id, sample_limit=5):

        super().__init__()

        self.cp = cp
        self.gpu_id = gpu_id
        self.sample_limit = sample_limit

        self.total_buffers = 0
        self.total_frames = 0

        self.frame_processing_times = []

    def consume(self, buffer):

        start_time = time.perf_counter()

        self.total_buffers += 1

        batch_meta = buffer.batch_meta

        frame_items = list(batch_meta.frame_items)

        if not frame_items:

            raise RuntimeError(
                "Received a buffer without frame metadata."
            )

        for frame_meta in frame_items:

            batch_id = int(frame_meta.batch_id)

            # ----------------------------------------
            # Extract actual frame tensor
            # ----------------------------------------

            ds_tensor = buffer.extract(batch_id)

            if not ds_tensor:

                raise RuntimeError(
                    f"Failed to extract frame {batch_id}."
                )

            # ----------------------------------------
            # Inspect DeepStream tensor
            # ----------------------------------------

            tensor_shape = tuple(ds_tensor.shape)

            tensor_dtype = str(ds_tensor.dtype)

            tensor_strides = tuple(ds_tensor.strides)

            tensor_device_type = ds_tensor.device_type

            dlpack_device = tuple(
                ds_tensor.__dlpack_device__()
            )

            dlpack_device_type, dlpack_device_id = (
                dlpack_device
            )

            if tensor_device_type != "GPU":

                raise RuntimeError(
                    "Expected GPU tensor, received "
                    f"{tensor_device_type}."
                )

            if dlpack_device_type != 2:

                raise RuntimeError(
                    "Expected DLPack CUDA device type 2, "
                    f"received {dlpack_device_type}."
                )

            if dlpack_device_id != self.gpu_id:

                raise RuntimeError(
                    f"Tensor belongs to GPU {dlpack_device_id}, "
                    f"expected GPU {self.gpu_id}."
                )

            # ----------------------------------------
            # Import into CuPy through DLPack
            # ----------------------------------------

            gpu_frame = self.cp.from_dlpack(
                ds_tensor
            )

            if gpu_frame.device.id != self.gpu_id:

                raise RuntimeError(
                    "CuPy frame is on the wrong GPU."
                )

            if tuple(gpu_frame.shape) != tensor_shape:

                raise RuntimeError(
                    "DeepStream and CuPy tensor shapes differ."
                )

            # ----------------------------------------
            # Perform an actual GPU memory operation
            # ----------------------------------------

            sample_count = min(
                1024,
                gpu_frame.size
            )

            gpu_result = self.cp.sum(
                gpu_frame.reshape(-1)[:sample_count],
                dtype=self.cp.uint64
            )

            # Complete GPU access before the original
            # DeepStream buffer is released.
            self.cp.cuda.get_current_stream().synchronize()

            # ----------------------------------------
            # Print a limited number of samples
            # ----------------------------------------

            if self.total_frames < self.sample_limit:

                print("\n" + "-" * 60)

                print(
                    f"FRAME {self.total_frames + 1}"
                )

                print("-" * 60)

                print(
                    f"Frame number: "
                    f"{int(frame_meta.frame_number)}"
                )

                print(
                    f"Source ID: "
                    f"{int(frame_meta.source_id)}"
                )

                print(
                    f"Batch ID: {batch_id}"
                )

                print(
                    f"Buffer PTS: "
                    f"{int(frame_meta.buffer_pts)}"
                )

                print("\nDeepStream Tensor:")

                print(
                    f"Shape: {tensor_shape}"
                )

                print(
                    f"Dtype: {tensor_dtype}"
                )

                print(
                    f"Strides: {tensor_strides}"
                )

                print(
                    f"Device type: {tensor_device_type}"
                )

                print(
                    f"DLPack device: {dlpack_device}"
                )

                print("\nCuPy GPU Array:")

                print(
                    f"Shape: {gpu_frame.shape}"
                )

                print(
                    f"Dtype: {gpu_frame.dtype}"
                )

                print(
                    f"GPU ID: {gpu_frame.device.id}"
                )

                print(
                    f"Memory size: "
                    f"{gpu_frame.nbytes / 1024**2:.3f} MiB"
                )

                print(
                    "GPU memory access: SUCCESS"
                )

            self.total_frames += 1

            # Keep the extracted tensor and GPU view alive
            # until all GPU operations have completed.
            del gpu_result
            del gpu_frame
            del ds_tensor

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        self.frame_processing_times.append(
            elapsed_ms
        )

        return 1

    def get_summary(self):

        return {

            "total_buffers": self.total_buffers,

            "total_frames": self.total_frames,

            "average_callback_ms": (
                statistics.mean(
                    self.frame_processing_times
                )
                if self.frame_processing_times
                else 0.0
            ),

            "max_callback_ms": (
                max(self.frame_processing_times)
                if self.frame_processing_times
                else 0.0
            ),
        }


def positive_int(value):

    number = int(value)

    if number <= 0:

        raise argparse.ArgumentTypeError(
            "Value must be greater than zero."
        )

    return number


def nonnegative_int(value):

    number = int(value)

    if number < 0:

        raise argparse.ArgumentTypeError(
            "Value cannot be negative."
        )

    return number


def main():

    parser = argparse.ArgumentParser(
        description="DeepStream GPU frame extraction test"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Video path inside the container"
    )

    parser.add_argument(
        "--gpu-id",
        type=nonnegative_int,
        default=0
    )

    parser.add_argument(
        "--sample-limit",
        type=nonnegative_int,
        default=5
    )

    args = parser.parse_args()

    video_path = Path(args.source)

    if not video_path.is_file():

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    import cupy as cp

    cp.cuda.Device(args.gpu_id).use()

    properties = (
        cp.cuda.runtime.getDeviceProperties(
            args.gpu_id
        )
    )

    gpu_name = properties["name"]

    if isinstance(gpu_name, bytes):

        gpu_name = gpu_name.decode()

    print("=" * 65)

    print(
        "DEEPSTREAM GPU FRAME EXTRACTION TEST"
    )

    print("=" * 65)

    print(f"Video: {video_path}")

    print(f"GPU: {gpu_name}")

    print(f"GPU ID: {args.gpu_id}")

    print(
        f"Pipeline resolution: "
        f"{BATCH_WIDTH}x{BATCH_HEIGHT}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    # --------------------------------------------
    # Initialize BufferRetriever
    # --------------------------------------------

    retriever = GPUFrameInspectionRetriever(
        cp=cp,
        gpu_id=args.gpu_id,
        sample_limit=args.sample_limit
    )

    # --------------------------------------------
    # Build DeepStream pipeline
    # --------------------------------------------

    pipeline = Pipeline(
        "deepstream-gpu-frame-inspection"
    )

    flow = (
        Flow(pipeline)

        .capture(
            [str(video_path)],
            gpu_id=args.gpu_id
        )

        .batch(
            batch_size=BATCH_SIZE,
            width=BATCH_WIDTH,
            height=BATCH_HEIGHT
        )

        .retrieve(retriever)
    )

    # --------------------------------------------
    # Run pipeline
    # --------------------------------------------

    print(
        "\nStarting DeepStream GPU frame extraction...\n"
    )

    start_time = time.perf_counter()

    flow()

    elapsed_seconds = (
        time.perf_counter() - start_time
    )

    # --------------------------------------------
    # Report results
    # --------------------------------------------

    summary = retriever.get_summary()

    print("\n" + "=" * 65)

    print(
        "GPU FRAME EXTRACTION STATISTICS"
    )

    print("=" * 65)

    print(
        f"Total buffers: "
        f"{summary['total_buffers']}"
    )

    print(
        f"Total frames: "
        f"{summary['total_frames']}"
    )

    print(
        f"Total elapsed seconds: "
        f"{elapsed_seconds:.3f}"
    )

    print(
        f"Average callback time: "
        f"{summary['average_callback_ms']:.3f} ms"
    )

    print(
        f"Maximum callback time: "
        f"{summary['max_callback_ms']:.3f} ms"
    )

    if elapsed_seconds > 0:

        print(
            f"Observed processing FPS: "
            f"{summary['total_frames'] / elapsed_seconds:.2f}"
        )

    free_bytes, total_bytes = (
        cp.cuda.runtime.memGetInfo()
    )

    print("\n========== GPU MEMORY ==========")

    print(
        f"Total: {total_bytes / 1024**2:.2f} MiB"
    )

    print(
        f"Free: {free_bytes / 1024**2:.2f} MiB"
    )

    print("\n" + "=" * 65)

    print(
        "DEEPSTREAM GPU FRAME EXTRACTION TEST PASSED"
    )

    print("=" * 65)


if __name__ == "__main__":

    sys.exit(main())
