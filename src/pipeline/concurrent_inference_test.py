"""
Concurrent inference test.

Architecture:
    One GPU frame pool
        |
        +----> Model A input queue
        |          |
        |          +----> GPU Worker A
        |                     |
        |                     +----> CUDA Stream A
        |                     |
        |                     +----> Model A
        |                     |
        |                     +----> Output Queue A
        |
        +----> Model B input queue
                   |
                   +----> GPU Worker B
                              |
                              +----> CUDA Stream B
                              |
                              +----> Model B
                              |
                              +----> Output Queue B

Both workers receive the same GPU frame and packet objects.

The models execute independently and publish independent results.

This is a single-process, multi-threaded concurrency experiment.
It does not yet connect to DeepStream or Kafka.
"""

import argparse
import queue
import statistics
import sys
import time

from src.models.mock_gpu_model_a import MockGPUModelA
from src.models.mock_gpu_model_b import MockGPUModelB
from src.workers.gpu_model_worker import GPUModelWorker


# ========================================================
# ARGUMENT VALIDATION
# ========================================================

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


# ========================================================
# STATISTICS
# ========================================================

def percentile(values, q):

    ordered = sorted(values)

    position = (len(ordered) - 1) * q / 100.0

    lower = int(position)

    upper = min(
        lower + 1,
        len(ordered) - 1
    )

    fraction = position - lower

    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def drain_queue(target_queue):

    items = []

    while True:

        try:

            items.append(
                target_queue.get_nowait()
            )

        except queue.Empty:

            break

    return items


# ========================================================
# GPU FRAME POOL
# ========================================================

def create_gpu_frame_pool(
    cp,
    width,
    height,
    pool_size,
    seed
):

    print("\n========== GPU FRAME POOL ==========")

    rng = cp.random.RandomState(seed)

    frames = []

    for index in range(pool_size):

        frame = rng.randint(
            0,
            256,
            size=(height, width, 3),
            dtype=cp.uint8
        )

        frames.append(frame)

        print(
            f"Created GPU frame {index + 1}/{pool_size}"
        )

    # Ensure all frames are ready before workers
    # begin accessing them through independent streams.
    cp.cuda.runtime.deviceSynchronize()

    total_bytes = sum(
        frame.nbytes for frame in frames
    )

    print(f"Frame count: {len(frames)}")

    print(
        f"Frame shape: {frames[0].shape}"
    )

    print(
        f"Frame dtype: {frames[0].dtype}"
    )

    print(
        f"Total frame pool memory: "
        f"{total_bytes / 1024**2:.2f} MiB"
    )

    return frames


# ========================================================
# FRAME DISTRIBUTION
# ========================================================

def distribute_frame(
    frame_id,
    frame_pool,
    input_queue_a,
    input_queue_b
):

    # Reuse a GPU frame from the pool.
    frame = frame_pool[
        frame_id % len(frame_pool)
    ]

    # Construct ONE packet.
    packet = {
        "frame_id": frame_id,
        "frame": frame
    }

    # Send the SAME packet object to both queues.
    input_queue_a.put(packet)

    input_queue_b.put(packet)


# ========================================================
# WORKER RESULT STATISTICS
# ========================================================

def print_worker_statistics(
    worker_name,
    results
):

    print(
        f"\n========== {worker_name} PERFORMANCE =========="
    )

    metrics = [
        "preprocessing_gpu_ms",
        "inference_gpu_ms",
        "postprocessing_gpu_ms",
        "gpu_total_ms",
        "output_transfer_ms",
        "cpu_result_build_ms",
        "end_to_end_ms"
    ]

    for metric in metrics:

        values = [
            item["result"]["timings_ms"][metric]
            for item in results
        ]

        print(f"\n{metric}")

        print(
            f"  Mean: {statistics.mean(values):.5f} ms"
        )

        print(
            f"  P50:  {percentile(values, 50):.5f} ms"
        )

        print(
            f"  P95:  {percentile(values, 95):.5f} ms"
        )

        print(
            f"  P99:  {percentile(values, 99):.5f} ms"
        )

        print(
            f"  Min:  {min(values):.5f} ms"
        )

        print(
            f"  Max:  {max(values):.5f} ms"
        )


# ========================================================
# SHARED FRAME VALIDATION
# ========================================================

def validate_shared_frames(
    results_a,
    results_b,
    expected_frames
):

    print("\n========== SHARED FRAME VALIDATION ==========")

    if len(results_a) != expected_frames:
        raise RuntimeError(
            f"Model A returned {len(results_a)} results; "
            f"expected {expected_frames}."
        )

    if len(results_b) != expected_frames:
        raise RuntimeError(
            f"Model B returned {len(results_b)} results; "
            f"expected {expected_frames}."
        )

    results_a_by_id = {
        item["frame_id"]: item
        for item in results_a
    }

    results_b_by_id = {
        item["frame_id"]: item
        for item in results_b
    }

    for frame_id in range(expected_frames):

        result_a = results_a_by_id.get(frame_id)

        result_b = results_b_by_id.get(frame_id)

        if result_a is None or result_b is None:

            raise RuntimeError(
                f"Missing result for frame {frame_id}."
            )

        # Both workers must receive the same packet.
        if (
            result_a["packet_identity"]
            != result_b["packet_identity"]
        ):

            raise RuntimeError(
                f"Packet identity mismatch for frame {frame_id}."
            )

        # Both workers must receive the same GPU array.
        if (
            result_a["frame_identity"]
            != result_b["frame_identity"]
        ):

            raise RuntimeError(
                f"Frame identity mismatch for frame {frame_id}."
            )

        # Verify independent model outputs.
        if (
            result_a["model_id"]
            != "mock_model_a"
        ):

            raise RuntimeError(
                "Unexpected Model A output."
            )

        if (
            result_b["model_id"]
            != "mock_model_b"
        ):

            raise RuntimeError(
                "Unexpected Model B output."
            )

    print(
        f"Verified frame pairs: {expected_frames}"
    )

    print(
        "Shared packet identity: SUCCESS"
    )

    print(
        "Shared GPU frame identity: SUCCESS"
    )

    print(
        "Independent model results: SUCCESS"
    )


# ========================================================
# MAIN
# ========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Concurrent GPU inference test"
    )

    parser.add_argument(
        "--gpu-id",
        type=nonnegative_int,
        default=0
    )

    parser.add_argument(
        "--frame-width",
        type=positive_int,
        default=1280
    )

    parser.add_argument(
        "--frame-height",
        type=positive_int,
        default=720
    )

    parser.add_argument(
        "--input-size",
        type=positive_int,
        default=320
    )

    parser.add_argument(
        "--frames",
        type=positive_int,
        default=100
    )

    parser.add_argument(
        "--warmup",
        type=nonnegative_int,
        default=10
    )

    parser.add_argument(
        "--frame-pool-size",
        type=positive_int,
        default=8
    )

    parser.add_argument(
        "--queue-size",
        type=positive_int,
        default=4
    )

    args = parser.parse_args()

    if args.input_size % 16 != 0:

        parser.error(
            "--input-size must be divisible by 16."
        )

    print("=" * 65)

    print(
        "CONCURRENT GPU INFERENCE TEST"
    )

    print("=" * 65)

    workers = []

    try:

        import cupy as cp

        # --------------------------------------------
        # GPU initialization
        # --------------------------------------------

        cp.cuda.Device(args.gpu_id).use()

        properties = (
            cp.cuda.runtime.getDeviceProperties(
                args.gpu_id
            )
        )

        gpu_name = properties["name"]

        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()

        print(f"\nGPU: {gpu_name}")

        print(f"GPU ID: {args.gpu_id}")

        # --------------------------------------------
        # Generate shared GPU frame pool
        # --------------------------------------------

        frame_pool = create_gpu_frame_pool(
            cp=cp,
            width=args.frame_width,
            height=args.frame_height,
            pool_size=args.frame_pool_size,
            seed=123
        )

        # --------------------------------------------
        # Initialize models
        # --------------------------------------------

        print("\n========== MODEL INITIALIZATION ==========")

        print("Initializing Model A...")

        model_a = MockGPUModelA(
            cp=cp,
            gpu_id=args.gpu_id,
            input_size=args.input_size,
            hidden_units=128,
            top_k=5,
            seed=42
        )

        print("Model A initialized.")

        print("Initializing Model B...")

        model_b = MockGPUModelB(
            cp=cp,
            gpu_id=args.gpu_id,
            input_size=args.input_size,
            hidden_units=512,
            top_k=5,
            seed=84
        )

        print("Model B initialized.")

        cp.cuda.runtime.deviceSynchronize()

        # --------------------------------------------
        # Independent queues
        # --------------------------------------------

        print("\n========== QUEUE INITIALIZATION ==========")

        input_queue_a = queue.Queue(
            maxsize=args.queue_size
        )

        input_queue_b = queue.Queue(
            maxsize=args.queue_size
        )

        # Output queues are unbounded in this test.
        # A production implementation should introduce
        # explicit output backpressure as well.
        output_queue_a = queue.Queue()

        output_queue_b = queue.Queue()

        error_queue = queue.Queue()

        print(
            f"Input queue A capacity: {args.queue_size}"
        )

        print(
            f"Input queue B capacity: {args.queue_size}"
        )

        print(
            "Output queues: Independent"
        )

        # --------------------------------------------
        # Create independent workers and streams
        # --------------------------------------------

        print("\n========== WORKER INITIALIZATION ==========")

        worker_a = GPUModelWorker(
            name="model-a-worker",
            model=model_a,
            cp=cp,
            gpu_id=args.gpu_id,
            input_queue=input_queue_a,
            output_queue=output_queue_a,
            error_queue=error_queue
        )

        worker_b = GPUModelWorker(
            name="model-b-worker",
            model=model_b,
            cp=cp,
            gpu_id=args.gpu_id,
            input_queue=input_queue_b,
            output_queue=output_queue_b,
            error_queue=error_queue
        )

        if worker_a.stream.ptr == worker_b.stream.ptr:

            raise RuntimeError(
                "Workers unexpectedly share a CUDA stream."
            )

        print(
            f"Model A CUDA stream: {worker_a.stream.ptr}"
        )

        print(
            f"Model B CUDA stream: {worker_b.stream.ptr}"
        )

        print(
            "Independent CUDA streams: VERIFIED"
        )

        # --------------------------------------------
        # Start workers
        # --------------------------------------------

        print("\n========== STARTING WORKERS ==========")

        worker_a.start()

        workers.append(worker_a)

        worker_b.start()

        workers.append(worker_b)

        # --------------------------------------------
        # Warm-up
        # --------------------------------------------

        print("\n========== WARM-UP ==========")

        print(
            f"Warm-up frames: {args.warmup}"
        )

        for frame_id in range(
            -args.warmup,
            0
        ):

            distribute_frame(
                frame_id=frame_id,
                frame_pool=frame_pool,
                input_queue_a=input_queue_a,
                input_queue_b=input_queue_b
            )

        # Wait until both workers finish warm-up.
        input_queue_a.join()

        input_queue_b.join()

        # Discard warm-up results.
        warmup_results_a = drain_queue(
            output_queue_a
        )

        warmup_results_b = drain_queue(
            output_queue_b
        )

        warmup_errors = drain_queue(
            error_queue
        )

        if warmup_errors:

            for error in warmup_errors:

                print(
                    error["traceback"]
                )

            raise RuntimeError(
                "Worker error during warm-up."
            )

        if (
            len(warmup_results_a) != args.warmup
            or len(warmup_results_b) != args.warmup
        ):

            raise RuntimeError(
                "Warm-up result count mismatch."
            )

        print("Warm-up completed successfully.")

        # --------------------------------------------
        # Concurrent inference benchmark
        # --------------------------------------------

        print("\n========== CONCURRENT INFERENCE ==========")

        print(
            f"Frames: {args.frames}"
        )

        print(
            "Execution mode: Two independent workers"
        )

        print(
            "Input: Shared GPU-resident frame objects"
        )

        benchmark_start = time.perf_counter_ns()

        # --------------------------------------------
        # Producer / frame distribution
        # --------------------------------------------

        for frame_id in range(args.frames):

            distribute_frame(
                frame_id=frame_id,
                frame_pool=frame_pool,
                input_queue_a=input_queue_a,
                input_queue_b=input_queue_b
            )

            if (frame_id + 1) % 20 == 0:

                print(
                    f"Submitted {frame_id + 1}/"
                    f"{args.frames} frames"
                )

        submission_end = time.perf_counter_ns()

        # --------------------------------------------
        # Wait for both workers
        # --------------------------------------------

        print("\nWaiting for Model A...")

        input_queue_a.join()

        print("Model A finished.")

        print("\nWaiting for Model B...")

        input_queue_b.join()

        print("Model B finished.")

        benchmark_end = time.perf_counter_ns()

        submission_time_ms = (
            submission_end - benchmark_start
        ) / 1_000_000

        total_time_ms = (
            benchmark_end - benchmark_start
        ) / 1_000_000

        # --------------------------------------------
        # Shutdown workers gracefully
        # --------------------------------------------

        print("\n========== WORKER SHUTDOWN ==========")

        for worker in workers:

            worker.stop()

        for worker in workers:

            stopped = worker.join(timeout=30)

            if not stopped:

                raise RuntimeError(
                    f"Worker {worker.name} failed to stop."
                )

        print("All workers stopped successfully.")

        # --------------------------------------------
        # Collect independent results
        # --------------------------------------------

        results_a = drain_queue(
            output_queue_a
        )

        results_b = drain_queue(
            output_queue_b
        )

        errors = drain_queue(
            error_queue
        )

        if errors:

            print("\n========== WORKER ERRORS ==========")

            for error in errors:

                print(
                    f"\nWorker: {error['worker_name']}"
                )

                print(
                    f"Frame: {error['frame_id']}"
                )

                print(
                    f"Error: {error['error']}"
                )

                print(
                    error["traceback"]
                )

            raise RuntimeError(
                "One or more GPU workers failed."
            )

        # --------------------------------------------
        # Verify shared GPU frame delivery
        # --------------------------------------------

        validate_shared_frames(
            results_a=results_a,
            results_b=results_b,
            expected_frames=args.frames
        )

        # --------------------------------------------
        # Individual worker statistics
        # --------------------------------------------

        print_worker_statistics(
            "MODEL A",
            results_a
        )

        print_worker_statistics(
            "MODEL B",
            results_b
        )

        # --------------------------------------------
        # Pipeline throughput
        # --------------------------------------------

        print("\n" + "=" * 65)

        print(
            "CONCURRENT PIPELINE PERFORMANCE"
        )

        print("=" * 65)

        print(
            f"Frames submitted: {args.frames}"
        )

        print(
            f"Submission time: {submission_time_ms:.3f} ms"
        )

        print(
            f"Total completion time: {total_time_ms:.3f} ms"
        )

        elapsed_seconds = total_time_ms / 1000.0

        frame_throughput = (
            args.frames / elapsed_seconds
        )

        inference_throughput = (
            2 * args.frames / elapsed_seconds
        )

        print(
            f"Camera frame throughput: "
            f"{frame_throughput:.2f} frames/sec"
        )

        print(
            f"Combined model invocation rate: "
            f"{inference_throughput:.2f} inferences/sec"
        )

        print(
            f"Model A completed: {len(results_a)}"
        )

        print(
            f"Model B completed: {len(results_b)}"
        )

        # --------------------------------------------
        # GPU memory
        # --------------------------------------------

        free_bytes, total_bytes = (
            cp.cuda.runtime.memGetInfo()
        )

        memory_pool = (
            cp.get_default_memory_pool()
        )

        print("\n========== GPU MEMORY ==========")

        print(
            f"Total GPU memory: "
            f"{total_bytes / 1024**2:.2f} MiB"
        )

        print(
            f"Free GPU memory: "
            f"{free_bytes / 1024**2:.2f} MiB"
        )

        print(
            f"CuPy memory pool: "
            f"{memory_pool.used_bytes() / 1024**2:.2f} MiB"
        )

        print("\n" + "=" * 65)

        print(
            "CONCURRENT INFERENCE TEST PASSED"
        )

        print("=" * 65)

        print(
            "\nNote: Independent CUDA streams enable "
            "concurrent scheduling, but this test does "
            "not prove that GPU kernels physically "
            "overlapped in time."
        )

        return 0

    except Exception as exc:

        print("\n" + "=" * 65)

        print("CONCURRENT INFERENCE TEST FAILED")

        print(f"Error: {exc}")

        print("=" * 65)

        return 1

    finally:

        # Ensure workers are stopped even if an
        # exception occurs during the experiment.
        for worker in workers:

            worker.stop()

        for worker in workers:

            if not worker.join(timeout=30):

                print(
                    f"WARNING: {worker.name} "
                    "did not stop cleanly."
                )


if __name__ == "__main__":

    sys.exit(main())
