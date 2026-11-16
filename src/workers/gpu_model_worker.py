
"""
Reusable GPU Model Worker.

Responsibilities:
    - Consume GPU-resident frames from an independent input queue.
    - Execute an assigned GPU model.
    - Use an independent CUDA stream.
    - Publish results to an independent output queue.
    - Report worker errors.
    - Support graceful shutdown.

This implementation uses Python threads within one process.

The same CuPy frame object can be submitted to multiple workers
without copying the full frame to CPU memory.
"""

import queue
import threading
import traceback


class GPUModelWorker:

    def __init__(
        self,
        name,
        model,
        cp,
        gpu_id,
        input_queue,
        output_queue,
        error_queue=None,
        stream=None
    ):

        self.name = name
        self.model = model
        self.cp = cp
        self.gpu_id = gpu_id

        self.input_queue = input_queue
        self.output_queue = output_queue

        self.error_queue = (
            error_queue
            if error_queue is not None
            else queue.Queue()
        )

        # Each worker owns an independent CUDA stream.
        with self.cp.cuda.Device(self.gpu_id):

            self.stream = (
                stream
                if stream is not None
                else self.cp.cuda.Stream(
                    non_blocking=True
                )
            )

        self._stop_token = object()

        self._thread = None

        self._stop_requested = False

        self._stop_lock = threading.Lock()

    # ----------------------------------------------------
    # START WORKER
    # ----------------------------------------------------

    def start(self):

        if self._thread is not None:

            raise RuntimeError(
                f"Worker {self.name} has already been started."
            )

        self._thread = threading.Thread(
            target=self._run,
            name=self.name,
            daemon=False
        )

        self._thread.start()

        return self

    # ----------------------------------------------------
    # WORKER LOOP
    # ----------------------------------------------------

    def _run(self):

        print(
            f"[{self.name}] Worker started "
            f"on GPU {self.gpu_id}."
        )

        while True:

            packet = self.input_queue.get()

            frame_id = None

            try:

                # ----------------------------------------
                # Shutdown signal
                # ----------------------------------------

                if packet is self._stop_token:

                    print(
                        f"[{self.name}] Shutdown signal received."
                    )

                    return

                # ----------------------------------------
                # Validate input packet
                # ----------------------------------------

                if not isinstance(packet, dict):

                    raise TypeError(
                        "Input packet must be a dictionary."
                    )

                if "frame" not in packet:

                    raise KeyError(
                        "Input packet is missing 'frame'."
                    )

                if "frame_id" not in packet:

                    raise KeyError(
                        "Input packet is missing 'frame_id'."
                    )

                frame = packet["frame"]

                frame_id = packet["frame_id"]

                if not isinstance(frame, self.cp.ndarray):

                    raise TypeError(
                        "Input frame must be a CuPy GPU array."
                    )

                if frame.device.id != self.gpu_id:

                    raise ValueError(
                        f"Frame belongs to GPU "
                        f"{frame.device.id}, but worker "
                        f"uses GPU {self.gpu_id}."
                    )

                # ----------------------------------------
                # Execute model on independent CUDA stream
                # ----------------------------------------

                with self.cp.cuda.Device(self.gpu_id):

                    with self.stream:

                        result = self.model.infer(
                            frame,
                            frame_id
                        )

                # ----------------------------------------
                # Independent result publishing
                # ----------------------------------------

                result_packet = {

                    "worker_name": self.name,

                    "model_id": result.get(
                        "model_id",
                        self.name
                    ),

                    "frame_id": frame_id,

                    # Used to verify shared object identity.
                    "packet_identity": id(packet),

                    "frame_identity": id(frame),

                    "result": result
                }

                self.output_queue.put(
                    result_packet
                )

            except Exception as exc:

                # Ensure pending operations on this
                # worker's stream are completed safely.
                try:

                    self.stream.synchronize()

                except Exception:

                    pass

                error_packet = {

                    "worker_name": self.name,

                    "frame_id": frame_id,

                    "error_type": type(exc).__name__,

                    "error": str(exc),

                    "traceback": traceback.format_exc()
                }

                self.error_queue.put(
                    error_packet
                )

                print(
                    f"[{self.name}] ERROR: {exc}"
                )

            finally:

                self.input_queue.task_done()

    # ----------------------------------------------------
    # STOP WORKER
    # ----------------------------------------------------

    def stop(self):

        if self._thread is None:

            return

        with self._stop_lock:

            if self._stop_requested:

                return

            self._stop_requested = True

            # FIFO queues ensure previously submitted
            # frames are processed before shutdown.
            self.input_queue.put(
                self._stop_token
            )

    # ----------------------------------------------------
    # WAIT FOR WORKER
    # ----------------------------------------------------

    def join(self, timeout=None):

        if self._thread is None:

            return True

        self._thread.join(timeout)

        return not self._thread.is_alive()

    # ----------------------------------------------------
    # WORKER STATUS
    # ----------------------------------------------------

    def is_alive(self):

        return (
            self._thread is not None
            and self._thread.is_alive()
        )
