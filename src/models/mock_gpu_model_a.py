
"""
Mock GPU Model A.

Synthetic GPU-based AI model for testing:
    - GPU-resident frame processing
    - GPU preprocessing
    - Neural-network-like computation
    - GPU postprocessing
    - CPU result materialization
    - Detailed performance instrumentation

No real AI model weights are required.

Input:
    CuPy GPU array: (height, width, 3), uint8

Output:
    Structured synthetic detection results and timings.

CUDA stream behavior:
    - Accepts an optional caller-provided CuPy CUDA stream.
    - Initializes weights and runs inference on that stream.
    - Does not modify the input frame.
"""

import argparse
import json
import statistics
import sys
import time


GRID_SIZE = 16
CANDIDATE_COUNT = 16

CLASS_NAMES = [
    "person",
    "vehicle",
    "object",
]


def positive_int(value):
    number = int(value)

    if number <= 0:
        raise argparse.ArgumentTypeError(
            "Value must be greater than zero."
        )

    return number


def create_synthetic_frame(cp, width, height, seed):
    """
    Generate an RGB frame directly in GPU memory.
    """

    rng = cp.random.RandomState(seed)

    return rng.randint(
        0,
        256,
        size=(height, width, 3),
        dtype=cp.uint8
    )


class MockGPUModelA:

    def __init__(
        self,
        cp,
        gpu_id=0,
        input_size=320,
        hidden_units=128,
        top_k=5,
        seed=42,
        stream=None
    ):

        self.cp = cp
        self.gpu_id = gpu_id
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.top_k = top_k

        if input_size % GRID_SIZE != 0:
            raise ValueError(
                f"input_size must be divisible by {GRID_SIZE}."
            )

        if not 1 <= top_k <= CANDIDATE_COUNT:
            raise ValueError(
                f"top_k must be between 1 and {CANDIDATE_COUNT}."
            )

        cp.cuda.Device(gpu_id).use()

        self.stream = (
            stream
            if stream is not None
            else cp.cuda.get_current_stream()
        )

        self.resize_cache = {}

        # ------------------------------------------------
        with self.stream:
            # Initialize synthetic neural network weights
            # ------------------------------------------------

            rng = cp.random.RandomState(seed)

            feature_count = GRID_SIZE * GRID_SIZE * 3

            second_hidden = max(32, hidden_units // 2)

            self.w1 = (
                rng.standard_normal(
                    (feature_count, hidden_units)
                ).astype(cp.float32)
                * cp.float32(
                    (2.0 / feature_count) ** 0.5
                )
            )

            self.b1 = cp.zeros(
                hidden_units,
                dtype=cp.float32
            )

            self.w2 = (
                rng.standard_normal(
                    (hidden_units, second_hidden)
                ).astype(cp.float32)
                * cp.float32(
                    (2.0 / hidden_units) ** 0.5
                )
            )

            self.b2 = cp.zeros(
                second_hidden,
                dtype=cp.float32
            )

            self.w3 = (
                rng.standard_normal(
                    (second_hidden, CANDIDATE_COUNT)
                ).astype(cp.float32)
                * cp.float32(
                    (1.0 / second_hidden) ** 0.5
                )
            )

            self.b3 = cp.zeros(
                CANDIDATE_COUNT,
                dtype=cp.float32
            )

    # ----------------------------------------------------
    # GPU PREPROCESSING
    # ----------------------------------------------------

    def preprocess(self, frame):

        cp = self.cp

        if not isinstance(frame, cp.ndarray):
            raise TypeError(
                "Input frame must be a CuPy GPU array."
            )

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "Expected RGB frame with shape (H, W, 3)."
            )

        if frame.dtype != cp.uint8:
            raise TypeError(
                "Expected uint8 input frame."
            )

        if frame.device.id != self.gpu_id:
            raise ValueError(
                f"Input frame is on GPU {frame.device.id}; "
                f"model is configured for GPU {self.gpu_id}."
            )

        height, width, _ = frame.shape

        cache_key = (height, width)

        if cache_key not in self.resize_cache:

            y_indices = cp.linspace(
                0,
                height - 1,
                self.input_size
            ).astype(cp.int32)

            x_indices = cp.linspace(
                0,
                width - 1,
                self.input_size
            ).astype(cp.int32)

            self.resize_cache[cache_key] = (
                y_indices,
                x_indices
            )

        y_indices, x_indices = (
            self.resize_cache[cache_key]
        )

        # Nearest-neighbor resizing on GPU.
        resized = frame[
            y_indices[:, None],
            x_indices[None, :],
            :
        ]

        # Normalize pixel values.
        normalized = (
            resized.astype(cp.float32)
            * cp.float32(1.0 / 255.0)
        )

        # GPU average pooling.
        block_size = self.input_size // GRID_SIZE

        pooled = normalized.reshape(
            GRID_SIZE,
            block_size,
            GRID_SIZE,
            block_size,
            3
        ).mean(axis=(1, 3))

        return pooled.reshape(1, -1)

    # ----------------------------------------------------
    # GPU INFERENCE
    # ----------------------------------------------------

    def forward(self, features):

        cp = self.cp

        hidden1 = cp.maximum(
            features @ self.w1 + self.b1,
            0
        )

        hidden2 = cp.maximum(
            hidden1 @ self.w2 + self.b2,
            0
        )

        logits = hidden2 @ self.w3 + self.b3

        scores = (
            cp.float32(1.0)
            / (
                cp.float32(1.0)
                + cp.exp(-logits)
            )
        )

        return scores

    # ----------------------------------------------------
    # GPU POSTPROCESSING
    # ----------------------------------------------------

    def postprocess_gpu(self, scores):

        cp = self.cp

        indices = cp.argsort(
            scores[0]
        )[-self.top_k:][::-1]

        selected_scores = scores[0][indices]

        grid_columns = (
            indices % 4
        ).astype(cp.float32)

        grid_rows = (
            indices // 4
        ).astype(cp.float32)

        x1 = grid_columns * 0.25 + 0.02
        y1 = grid_rows * 0.25 + 0.02

        box_size = (
            0.08 + selected_scores * 0.12
        )

        x2 = cp.minimum(x1 + box_size, 1.0)
        y2 = cp.minimum(y1 + box_size, 1.0)

        # Pack everything into one GPU array.
        # This allows one compact GPU-to-CPU transfer.
        packed = cp.stack(
            [
                indices.astype(cp.float32),
                selected_scores,
                x1,
                y1,
                x2,
                y2
            ],
            axis=1
        )

        return packed

    # ----------------------------------------------------
    # CPU RESULT MATERIALIZATION
    # ----------------------------------------------------

    def materialize_detections(self, packed_cpu):

        detections = []

        for row in packed_cpu:

            candidate_id = int(row[0])

            class_id = (
                candidate_id % len(CLASS_NAMES)
            )

            detections.append({
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "confidence": round(
                    float(row[1]), 5
                ),
                "bbox_normalized": [
                    round(float(value), 5)
                    for value in row[2:6]
                ],
                "synthetic": True
            })

        return detections

    # ----------------------------------------------------
    # COMPLETE INFERENCE WITH DETAILED TIMING
    # ----------------------------------------------------

    def infer(self, frame, frame_id):

        with self.cp.cuda.Device(self.gpu_id):
            with self.stream:
                return self._infer_on_stream(frame, frame_id)

    def _infer_on_stream(self, frame, frame_id):

        cp = self.cp

        end_to_end_start = time.perf_counter_ns()

        # CUDA events for each GPU stage.
        pre_start = cp.cuda.Event()
        pre_end = cp.cuda.Event()

        infer_start = cp.cuda.Event()
        infer_end = cp.cuda.Event()

        post_start = cp.cuda.Event()
        post_end = cp.cuda.Event()

        # --------------------------------------------
        # Stage 1: GPU preprocessing
        # --------------------------------------------

        pre_start.record(self.stream)

        features = self.preprocess(frame)

        pre_end.record(self.stream)

        # --------------------------------------------
        # Stage 2: GPU neural-network computation
        # --------------------------------------------

        infer_start.record(self.stream)

        scores = self.forward(features)

        infer_end.record(self.stream)

        # --------------------------------------------
        # Stage 3: GPU postprocessing
        # --------------------------------------------

        post_start.record(self.stream)

        packed = self.postprocess_gpu(scores)

        post_end.record(self.stream)

        # Wait once for all GPU stages to finish.
        post_end.synchronize()

        # --------------------------------------------
        # Retrieve GPU execution times
        # --------------------------------------------

        preprocessing_gpu_ms = cp.cuda.get_elapsed_time(
            pre_start,
            pre_end
        )

        inference_gpu_ms = cp.cuda.get_elapsed_time(
            infer_start,
            infer_end
        )

        postprocessing_gpu_ms = cp.cuda.get_elapsed_time(
            post_start,
            post_end
        )

        gpu_total_ms = (
            preprocessing_gpu_ms
            + inference_gpu_ms
            + postprocessing_gpu_ms
        )

        # --------------------------------------------
        # Stage 4: GPU-to-CPU output transfer
        # --------------------------------------------

        transfer_start = time.perf_counter_ns()

        packed_cpu = cp.asnumpy(
            packed,
            stream=self.stream,
            blocking=True
        )

        transfer_end = time.perf_counter_ns()

        output_transfer_ms = (
            transfer_end - transfer_start
        ) / 1_000_000

        # --------------------------------------------
        # Stage 5: CPU result construction
        # --------------------------------------------

        cpu_build_start = time.perf_counter_ns()

        detections = self.materialize_detections(
            packed_cpu
        )

        cpu_build_end = time.perf_counter_ns()

        cpu_result_build_ms = (
            cpu_build_end - cpu_build_start
        ) / 1_000_000

        # --------------------------------------------
        # End-to-end measurement
        # --------------------------------------------

        end_to_end_ms = (
            time.perf_counter_ns()
            - end_to_end_start
        ) / 1_000_000

        return {
            "model_id": "mock_model_a",
            "frame_id": int(frame_id),
            "gpu_id": self.gpu_id,

            "input_frame_shape": list(frame.shape),

            "model_input_size": self.input_size,

            "timings_ms": {
                "preprocessing_gpu_ms": round(
                    preprocessing_gpu_ms, 5
                ),

                "inference_gpu_ms": round(
                    inference_gpu_ms, 5
                ),

                "postprocessing_gpu_ms": round(
                    postprocessing_gpu_ms, 5
                ),

                "gpu_total_ms": round(
                    gpu_total_ms, 5
                ),

                "output_transfer_ms": round(
                    output_transfer_ms, 5
                ),

                "cpu_result_build_ms": round(
                    cpu_result_build_ms, 5
                ),

                "end_to_end_ms": round(
                    end_to_end_ms, 5
                )
            },

            "detection_count": len(detections),

            "detections": detections
        }


# ========================================================
# COMMAND LINE TEST
# ========================================================

def main():

    parser = argparse.ArgumentParser(
        description="Mock GPU Model A"
    )

    parser.add_argument(
        "--gpu-id",
        type=int,
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
        "--hidden-units",
        type=positive_int,
        default=128
    )

    parser.add_argument(
        "--top-k",
        type=positive_int,
        default=5
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5
    )

    parser.add_argument(
        "--iterations",
        type=positive_int,
        default=20
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    if args.gpu_id < 0:
        parser.error("--gpu-id cannot be negative")

    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    if args.input_size % GRID_SIZE != 0:
        parser.error(
            f"--input-size must be divisible by {GRID_SIZE}"
        )

    if args.top_k > CANDIDATE_COUNT:
        parser.error(
            f"--top-k cannot exceed {CANDIDATE_COUNT}"
        )

    print("=" * 60)
    print("MOCK GPU MODEL A")
    print("=" * 60)

    try:
        import cupy as cp

    except ImportError as exc:
        print(f"CuPy import failed: {exc}")
        return 1

    try:

        cp.cuda.Device(args.gpu_id).use()

        properties = (
            cp.cuda.runtime.getDeviceProperties(
                args.gpu_id
            )
        )

        gpu_name = properties["name"]

        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()

        print(f"GPU: {gpu_name}")

        # --------------------------------------------
        # Generate synthetic GPU frame
        # --------------------------------------------

        print("\nGenerating synthetic GPU frame...")

        frame = create_synthetic_frame(
            cp,
            args.frame_width,
            args.frame_height,
            args.seed
        )

        cp.cuda.runtime.deviceSynchronize()

        print(f"Frame shape: {frame.shape}")
        print(f"Frame dtype: {frame.dtype}")

        # --------------------------------------------
        # Initialize model
        # --------------------------------------------

        print("\nInitializing Model A...")

        initialization_start = time.perf_counter()

        model = MockGPUModelA(
            cp=cp,
            gpu_id=args.gpu_id,
            input_size=args.input_size,
            hidden_units=args.hidden_units,
            top_k=args.top_k,
            seed=args.seed
        )

        cp.cuda.runtime.deviceSynchronize()

        initialization_time = (
            time.perf_counter() - initialization_start
        ) * 1000

        print(
            f"Model initialized in "
            f"{initialization_time:.3f} ms"
        )

        # --------------------------------------------
        # Warm-up
        # --------------------------------------------

        print("\n========== WARM-UP ==========")

        for i in range(args.warmup):

            model.infer(
                frame,
                frame_id=-(i + 1)
            )

        print("Warm-up completed.")

        # --------------------------------------------
        # Benchmark
        # --------------------------------------------

        print("\n========== INFERENCE BENCHMARK ==========")

        timing_keys = [
            "preprocessing_gpu_ms",
            "inference_gpu_ms",
            "postprocessing_gpu_ms",
            "gpu_total_ms",
            "output_transfer_ms",
            "cpu_result_build_ms",
            "end_to_end_ms"
        ]

        timing_samples = {
            key: [] for key in timing_keys
        }

        sample_result = None

        for i in range(args.iterations):

            result = model.infer(
                frame,
                frame_id=i
            )

            for key in timing_keys:

                timing_samples[key].append(
                    result["timings_ms"][key]
                )

            if sample_result is None:
                sample_result = result

        print(f"Iterations: {args.iterations}")

        print("\nAverage stage timings:")

        for key in timing_keys:

            average = statistics.mean(
                timing_samples[key]
            )

            print(
                f"{key}: {average:.5f} ms"
            )

        print("\nEnd-to-end latency range:")

        print(
            f"Minimum: "
            f"{min(timing_samples['end_to_end_ms']):.5f} ms"
        )

        print(
            f"Maximum: "
            f"{max(timing_samples['end_to_end_ms']):.5f} ms"
        )

        # --------------------------------------------
        # Sample result
        # --------------------------------------------

        print("\n========== SAMPLE RESULT ==========")

        print(
            json.dumps(
                sample_result,
                indent=2
            )
        )

        # --------------------------------------------
        # GPU memory
        # --------------------------------------------

        free_bytes, total_bytes = (
            cp.cuda.runtime.memGetInfo()
        )

        pool_bytes = (
            cp.get_default_memory_pool().used_bytes()
        )

        print("\n========== GPU MEMORY ==========")

        print(
            f"Total: {total_bytes / 1024**2:.2f} MiB"
        )

        print(
            f"Free: {free_bytes / 1024**2:.2f} MiB"
        )

        print(
            f"CuPy pool: {pool_bytes / 1024**2:.2f} MiB"
        )

        print("\n" + "=" * 60)
        print("MOCK GPU MODEL A TEST PASSED")
        print("=" * 60)

        return 0

    except Exception as exc:

        print("\nMODEL EXECUTION FAILED")
        print(f"Error: {exc}")

        return 1


if __name__ == "__main__":
    sys.exit(main())
