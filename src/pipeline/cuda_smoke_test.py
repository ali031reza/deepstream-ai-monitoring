"""
CUDA GPU Computation Smoke Test.

Purpose:
    Verify that CuPy can allocate GPU memory and execute
    actual CUDA computations inside the DeepStream container.

This test does not use real AI models or video frames.
"""

import argparse
import sys
import numpy as np


def positive_int(value: str) -> int:
    """Validate positive integer CLI arguments."""
    number = int(value)

    if number <= 0:
        raise argparse.ArgumentTypeError(
            "Value must be greater than zero."
        )

    return number


def format_cuda_version(version: int) -> str:
    """Convert CUDA integer version into readable format."""
    major = version // 1000
    minor = (version % 1000) // 10

    return f"{major}.{minor}"


def main() -> int:

    parser = argparse.ArgumentParser(
        description="DeepStream CUDA computation smoke test"
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="GPU device index (default: 0)"
    )

    parser.add_argument(
        "--elements",
        type=positive_int,
        default=1_000_000,
        help="Number of array elements (default: 1000000)"
    )

    parser.add_argument(
        "--repeats",
        type=positive_int,
        default=100,
        help="Number of benchmark iterations (default: 100)"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DEEPSTREAM CUDA COMPUTATION SMOKE TEST")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Import GPU libraries
    # --------------------------------------------------

    try:
        import cupy as cp

    except Exception as exc:
        print(f"ERROR: Could not import CuPy: {exc}")
        return 1

    print(f"CuPy version: {cp.__version__}")

    # --------------------------------------------------
    # 2. Detect GPU
    # --------------------------------------------------

    try:
        device_count = cp.cuda.runtime.getDeviceCount()

        if device_count == 0:
            raise RuntimeError("No CUDA devices detected.")

        if args.gpu_id < 0 or args.gpu_id >= device_count:
            raise ValueError(
                f"GPU {args.gpu_id} unavailable. "
                f"Detected devices: {device_count}"
            )

        cp.cuda.Device(args.gpu_id).use()

        properties = cp.cuda.runtime.getDeviceProperties(
            args.gpu_id
        )

        gpu_name = properties["name"]

        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8")

        compute_capability = (
            f'{properties["major"]}.'
            f'{properties["minor"]}'
        )

        print("\n========== GPU INFORMATION ==========")
        print(f"GPU: {gpu_name}")
        print(f"GPU ID: {args.gpu_id}")
        print(f"Compute capability: {compute_capability}")

        runtime_version = (
            cp.cuda.runtime.runtimeGetVersion()
        )

        driver_version = (
            cp.cuda.runtime.driverGetVersion()
        )

        print(
            "CUDA runtime:",
            format_cuda_version(runtime_version)
        )

        print(
            "CUDA driver API:",
            format_cuda_version(driver_version)
        )

    except Exception as exc:
        print(f"ERROR: GPU initialization failed: {exc}")
        return 1

    # --------------------------------------------------
    # 3. Inspect GPU memory
    # --------------------------------------------------

    try:
        free_bytes, total_bytes = (
            cp.cuda.runtime.memGetInfo()
        )

        print("\n========== GPU MEMORY ==========")

        print(
            f"Total GPU memory: "
            f"{total_bytes / (1024 ** 2):.2f} MiB"
        )

        print(
            f"Free GPU memory: "
            f"{free_bytes / (1024 ** 2):.2f} MiB"
        )

    except Exception as exc:
        print(f"ERROR: GPU memory inspection failed: {exc}")
        return 1

    # --------------------------------------------------
    # 4. Allocate GPU arrays
    # --------------------------------------------------

    try:
        print("\n========== GPU ALLOCATION ==========")

        print(f"Array elements: {args.elements:,}")

        print("Allocating input arrays...")

        # These arrays are allocated in GPU memory.
        a = cp.arange(
            args.elements,
            dtype=cp.float32
        )

        b = cp.full(
            args.elements,
            2.0,
            dtype=cp.float32
        )

        c = cp.empty_like(a)

        cp.cuda.runtime.deviceSynchronize()

        allocated_bytes = (
            cp.get_default_memory_pool().used_bytes()
        )

        print(
            f"CuPy memory pool usage: "
            f"{allocated_bytes / (1024 ** 2):.2f} MiB"
        )

    except Exception as exc:
        print(f"ERROR: GPU memory allocation failed: {exc}")
        return 1

    # --------------------------------------------------
    # 5. Execute actual GPU computation
    # --------------------------------------------------

    try:
        print("\n========== GPU COMPUTATION ==========")

        print("Operation: C = A + B")
        print("Performing warm-up...")

        for _ in range(10):
            cp.add(a, b, out=c)

        cp.cuda.runtime.deviceSynchronize()

        print("Warm-up completed.")

        # CUDA events measure execution on the GPU.
        start_event = cp.cuda.Event()
        end_event = cp.cuda.Event()

        print(f"Running {args.repeats} iterations...")

        start_event.record()

        for _ in range(args.repeats):
            cp.add(a, b, out=c)

        end_event.record()
        end_event.synchronize()

        total_gpu_time_ms = cp.cuda.get_elapsed_time(
            start_event,
            end_event
        )

        average_gpu_time_ms = (
            total_gpu_time_ms / args.repeats
        )

        print("GPU computation completed.")

        print(
            f"Total GPU execution time: "
            f"{total_gpu_time_ms:.4f} ms"
        )

        print(
            f"Average GPU execution time: "
            f"{average_gpu_time_ms:.6f} ms"
        )

    except Exception as exc:
        print(f"ERROR: GPU computation failed: {exc}")
        return 1

    # --------------------------------------------------
    # 6. Verify GPU output against CPU calculation
    # --------------------------------------------------

    try:
        print("\n========== RESULT VALIDATION ==========")

        print("Copying final result to CPU...")

        actual = cp.asnumpy(c)

        expected = (
            np.arange(
                args.elements,
                dtype=np.float32
            )
            + np.float32(2.0)
        )

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-6,
            atol=1e-6
        )

        max_error = float(
            np.max(np.abs(actual - expected))
        )

        print("GPU result matches CPU reference.")
        print(f"Maximum numerical error: {max_error}")

        print("\nSample results:")

        for index in range(min(5, args.elements)):
            print(
                f"A[{index}] = {actual[index] - 2.0:.1f}, "
                f"B[{index}] = 2.0, "
                f"C[{index}] = {actual[index]:.1f}"
            )

    except Exception as exc:
        print(f"ERROR: Result validation failed: {exc}")
        return 1

    # --------------------------------------------------
    # 7. Final report
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("CUDA COMPUTATION TEST PASSED")
    print("=" * 60)

    print("GPU initialization: SUCCESS")
    print("GPU memory allocation: SUCCESS")
    print("GPU computation: SUCCESS")
    print("CPU/GPU result validation: SUCCESS")

    print("\nOur Python environment can execute CUDA computations.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
