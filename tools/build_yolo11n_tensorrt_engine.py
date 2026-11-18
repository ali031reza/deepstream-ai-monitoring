"""Build a static YOLO11n FP16 TensorRT engine using the container's TRT runtime.

Run inside the DeepStream 9.2 container on the target GPU. TensorRT engines
are runtime/GPU specific; do not build on another machine and copy the plan.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

import tensorrt as trt

from configs.settings import load_settings


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        default=str(settings.paths.resolve("artifacts/models/yolo11n.onnx")),
    )
    parser.add_argument(
        "--engine",
        default=str(settings.paths.resolve("artifacts/models/yolo11n_fp16.engine")),
    )
    parser.add_argument("--workspace-gib", type=float, default=2.0)
    args = parser.parse_args()

    onnx_path = settings.paths.resolve(args.onnx)
    engine_path = settings.paths.resolve(args.engine)
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if args.workspace_gib <= 0:
        parser.error("--workspace-gib must be positive")

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    if not builder.platform_has_fast_fp16:
        raise RuntimeError("This GPU does not report fast FP16 support")

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    onnx_parser = trt.OnnxParser(network, logger)
    if not onnx_parser.parse(onnx_path.read_bytes()):
        errors = [
            str(onnx_parser.get_error(index))
            for index in range(onnx_parser.num_errors)
        ]
        raise RuntimeError("TensorRT ONNX parsing failed:\n" + "\n".join(errors))

    if network.num_inputs != 1 or network.num_outputs != 1:
        raise RuntimeError(
            f"Expected one input and one output; found "
            f"{network.num_inputs} input(s), {network.num_outputs} output(s)"
        )

    input_tensor = network.get_input(0)
    expected_shape = (1, 3, settings.yolo.image_size, settings.yolo.image_size)
    if tuple(input_tensor.shape) != expected_shape:
        raise RuntimeError(
            f"ONNX input is {tuple(input_tensor.shape)}; expected {expected_shape}. "
            "Export a static batch-1 ONNX file at the configured image size."
        )

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(args.workspace_gib * (1024 ** 3)),
    )
    config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the FP16 engine")

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{engine_path.name}.",
        suffix=".tmp",
        dir=engine_path.parent,
        delete=False,
    ) as temporary:
        temporary.write(bytes(serialized))
        temporary.flush()
        temporary_path = Path(temporary.name)

    temporary_path.replace(engine_path)

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        engine_path.unlink(missing_ok=True)
        raise RuntimeError("Built engine could not be deserialized")

    print(f"TensorRT version: {trt.__version__}")
    print(f"Engine written: {engine_path}")
    print(f"Input: {input_tensor.name} {tuple(input_tensor.shape)}")
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        print(
            f"Tensor {name}: mode={engine.get_tensor_mode(name)} "
            f"shape={tuple(engine.get_tensor_shape(name))} "
            f"dtype={engine.get_tensor_dtype(name)}"
        )
    print("YOLO11n TensorRT FP16 engine build and deserialize check PASSED")


if __name__ == "__main__":
    main()
