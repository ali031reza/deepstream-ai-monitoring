
"""
DeepStream BufferRetriever + CuPy DLPack Integration Test.

Objectives:
    1. Decode a video using NVIDIA DeepStream.
    2. Retrieve actual decoded frame tensors.
    3. Inspect CUDA device and tensor properties.
    4. Import DeepStream tensors into CuPy through DLPack.
    5. Verify shape, dtype and GPU identity.
    6. Execute a real GPU computation.
    7. Synchronize GPU work before returning from the callback.

Important:
    This is a synchronous interoperability test.

    GPU operations are completed inside consume(), while the
    DeepStream buffer is still valid.

    Asynchronous worker sharing will be implemented separately
    after we establish safe buffer ownership and lifetime.
"""

import argparse
import traceback
from pathlib import Path

import cupy as cp

from pyservicemaker import (
    Flow,
    Pipeline,
    BufferRetriever,
)


CUDA_DLPACK_DEVICE_TYPE = 2


def nonnegative_int(value):

    number = int(value)

    if number < 0:
        raise argparse.ArgumentTypeError(
            "Value cannot be negative."
        )

    return number


def positive_int(value):

    number = int(value)

    if number <= 0:
        raise argparse.ArgumentTypeError(
            "Value must be greater than zero."
        )

    return number


class DiagnosticRetriever(BufferRetriever):

    def __init__(
        self,
        gpu_id=0,
        sample_limit=5,
    ):

        super().__init__()

        self.gpu_id = gpu_id
        self.sample_limit = sample_limit

        self.count = 0
        self.extracted_frames = 0
        self.cupy_tests = 0
        self.errors = 0

        self.total_gpu_checksum = 0

    def consume(self, buffer):

        self.count += 1

        if (
            self.count <= self.sample_limit
            or self.count % 100 == 0
        ):

            print(
                f"\n[CALLBACK] Received buffer #{self.count}",
                flush=True
            )

        try:

            # ----------------------------------------
            # 1. Extract actual DeepStream frame tensor
            # ----------------------------------------

            tensor = buffer.extract(0)

            if tensor is None:

                raise RuntimeError(
                    "DeepStream returned an empty tensor."
                )

            shape = tuple(tensor.shape)

            dtype = str(tensor.dtype)

            device = tuple(
                tensor.__dlpack_device__()
            )

            print(
                f"[EXTRACT SUCCESS] "
                f"Shape={shape}, Dtype={dtype}"
                if self.count <= self.sample_limit
                else "",
                flush=True
            )

            # ----------------------------------------
            # 2. Validate CUDA device
            # ----------------------------------------

            if len(device) != 2:

                raise RuntimeError(
                    f"Unexpected DLPack device: {device}"
                )

            device_type, device_id = device

            if device_type != CUDA_DLPACK_DEVICE_TYPE:

                raise RuntimeError(
                    "Expected CUDA DLPack device type 2, "
                    f"received {device_type}."
                )

            if device_id != self.gpu_id:

                raise RuntimeError(
                    f"Tensor belongs to GPU {device_id}, "
                    f"expected GPU {self.gpu_id}."
                )

            self.extracted_frames += 1

            # ----------------------------------------
            # 3. Perform CuPy interoperability test
            #    on the first N frames
            # ----------------------------------------

            if self.cupy_tests < self.sample_limit:

                print(
                    "\n[CUPY] Importing DeepStream tensor "
                    "through DLPack...",
                    flush=True
                )

                with cp.cuda.Device(device_id):

                    # Import the existing CUDA tensor.
                    # No explicit CPU conversion is performed.
                    gpu_frame = cp.from_dlpack(tensor)

                    # --------------------------------
                    # 4. Validate imported GPU array
                    # --------------------------------

                    cupy_shape = tuple(gpu_frame.shape)

                    cupy_dtype = str(gpu_frame.dtype)

                    cupy_gpu_id = gpu_frame.device.id

                    if cupy_shape != shape:

                        raise RuntimeError(
                            f"Shape mismatch: "
                            f"{cupy_shape} != {shape}"
                        )

                    if cupy_dtype != dtype:

                        raise RuntimeError(
                            f"Dtype mismatch: "
                            f"{cupy_dtype} != {dtype}"
                        )

                    if cupy_gpu_id != device_id:

                        raise RuntimeError(
                            f"GPU mismatch: "
                            f"{cupy_gpu_id} != {device_id}"
                        )

                    # --------------------------------
                    # 5. Execute real GPU computation
                    # --------------------------------

                    # Calculate the sum of all frame pixels.
                    # Only the final scalar is transferred to CPU.

                    checksum_gpu = cp.sum(
                        gpu_frame,
                        dtype=cp.uint64
                    )

                    # Synchronize before returning from callback.
                    cp.cuda.runtime.deviceSynchronize()

                    checksum = int(
                        checksum_gpu.item()
                    )

                    self.total_gpu_checksum += checksum

                    # --------------------------------
                    # 6. Report successful operation
                    # --------------------------------

                    print(
                        "\n========== CUPY VALIDATION =========="
                    )

                    print(
                        f"Frame number: {self.count}"
                    )

                    print(
                        f"DeepStream shape: {shape}"
                    )

                    print(
                        f"CuPy shape: {cupy_shape}"
                    )

                    print(
                        f"DeepStream dtype: {dtype}"
                    )

                    print(
                        f"CuPy dtype: {cupy_dtype}"
                    )

                    print(
                        f"DLPack device: {device}"
                    )

                    print(
                        f"CuPy GPU ID: {cupy_gpu_id}"
                    )

                    print(
                        f"GPU checksum: {checksum}"
                    )

                    print(
                        "GPU computation: SUCCESS"
                    )

                    print(
                        "GPU synchronization: SUCCESS"
                    )

                    print(
                        "===================================="
                    )

                    self.cupy_tests += 1

                    # Release our references before callback returns.
                    del checksum_gpu
                    del gpu_frame

            # ----------------------------------------
            # 7. Return successfully
            # ----------------------------------------

            del tensor

            return 1

        except Exception as exc:

            self.errors += 1

            print(
                f"\n[ERROR] Buffer processing failed: {exc}",
                flush=True
            )

            traceback.print_exc()

            return -1

    def get_summary(self):

        return {

            "total_buffers": self.count,

            "extracted_frames": self.extracted_frames,

            "cupy_tests": self.cupy_tests,

            "errors": self.errors,

            "total_gpu_checksum": self.total_gpu_checksum,
        }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "DeepStream BufferRetriever and "
            "CuPy DLPack integration test"
        )
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Video path inside the container"
    )

    parser.add_argument(
        "--gpu-id",
        type=nonnegative_int,
        default=0,
        help="NVIDIA GPU ID"
    )

    parser.add_argument(
        "--sample-limit",
        type=positive_int,
        default=5,
        help="Number of frames used for CuPy testing"
    )

    args = parser.parse_args()

    video_path = Path(args.source)

    if not video_path.is_file():

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    # --------------------------------------------
    # 1. Initialize CUDA
    # --------------------------------------------

    gpu_count = cp.cuda.runtime.getDeviceCount()

    if args.gpu_id >= gpu_count:

        raise RuntimeError(
            f"GPU {args.gpu_id} does not exist. "
            f"Available GPUs: {gpu_count}"
        )

    cp.cuda.Device(args.gpu_id).use()

    properties = (
        cp.cuda.runtime.getDeviceProperties(
            args.gpu_id
        )
    )

    gpu_name = properties["name"]

    if isinstance(gpu_name, bytes):

        gpu_name = gpu_name.decode()

    # --------------------------------------------
    # 2. Print environment
    # --------------------------------------------

    print("=" * 65)

    print(
        "DEEPSTREAM + CUPY DLPACK INTEGRATION TEST"
    )

    print("=" * 65)

    print(f"Video: {video_path}")

    print(f"CuPy version: {cp.__version__}")

    print(f"GPU: {gpu_name}")

    print(f"GPU ID: {args.gpu_id}")

    print(f"CUDA runtime: {cp.cuda.runtime.runtimeGetVersion()}")

    print(f"Sample limit: {args.sample_limit}")

    # --------------------------------------------
    # 3. Initialize BufferRetriever
    # --------------------------------------------

    retriever = DiagnosticRetriever(
        gpu_id=args.gpu_id,
        sample_limit=args.sample_limit
    )

    # --------------------------------------------
    # 4. Build DeepStream pipeline
    # --------------------------------------------

    print(
        "\nBuilding DeepStream pipeline...",
        flush=True
    )

    flow = (
        Flow(
            Pipeline("retrieve-cupy-diagnostic")
        )

        .capture(
            [str(video_path)],
            gpu_id=args.gpu_id
        )

        .retrieve(
            retriever
        )
    )

    # --------------------------------------------
    # 5. Execute pipeline
    # --------------------------------------------

    print(
        "\nStarting DeepStream pipeline...",
        flush=True
    )

    flow()

    # --------------------------------------------
    # 6. Print results
    # --------------------------------------------

    summary = retriever.get_summary()

    print("\n" + "=" * 65)

    print(
        "DLPACK + CUPY INTEGRATION RESULTS"
    )

    print("=" * 65)

    print(
        f"Retrieved buffers: "
        f"{summary['total_buffers']}"
    )

    print(
        f"Extracted frames: "
        f"{summary['extracted_frames']}"
    )

    print(
        f"Successful CuPy tests: "
        f"{summary['cupy_tests']}"
    )

    print(
        f"Processing errors: "
        f"{summary['errors']}"
    )

    print(
        f"Combined sample checksum: "
        f"{summary['total_gpu_checksum']}"
    )

    # --------------------------------------------
    # 7. Validate final results
    # --------------------------------------------

    if summary["total_buffers"] == 0:

        raise RuntimeError(
            "No buffers were received."
        )

    if summary["errors"] > 0:

        raise RuntimeError(
            "One or more frames failed processing."
        )

    if summary["cupy_tests"] != min(
        args.sample_limit,
        summary["extracted_frames"]
    ):

        raise RuntimeError(
            "Not all requested CuPy tests completed."
        )

    print("\n" + "=" * 65)

    print(
        "DEEPSTREAM + CUPY DLPACK TEST PASSED"
    )

    print("=" * 65)


if __name__ == "__main__":

    main()
