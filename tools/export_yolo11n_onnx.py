"""Export the configured YOLO11n .pt weights to a static batch-1 ONNX model.

Run this in the existing WSL YOLO virtual environment, not in the
DeepStream container. The generated ONNX file is then consumed by the
TensorRT 10.16 engine builder inside the DeepStream 9.2 container.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from ultralytics import YOLO

from configs.settings import load_settings


def main() -> None:
    settings = load_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=str(settings.yolo.weights))
    parser.add_argument("--output", default=None)
    parser.add_argument("--image-size", type=int, default=settings.yolo.image_size)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    weights = settings.paths.resolve(args.weights)
    if not weights.is_file():
        raise FileNotFoundError(
            f"YOLO weights not found: {weights}. Run the existing YOLO11n "
            "video test once to download yolo11n.pt, or download the weights."
        )
    if args.image_size <= 0 or args.opset < 13:
        parser.error("--image-size must be positive and --opset must be >= 13")

    output = (
        settings.paths.resolve(args.output)
        if args.output
        else weights.with_name("yolo11n.onnx")
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    exported = model.export(
        format="onnx",
        imgsz=args.image_size,
        batch=1,
        dynamic=False,
        simplify=False,
        opset=args.opset,
        half=False,
        nms=False,
        device="cpu",
    )
    exported_path = Path(exported).expanduser().resolve()
    if not exported_path.is_file():
        raise RuntimeError(f"Ultralytics did not create ONNX output: {exported_path}")
    if exported_path != output:
        shutil.copy2(exported_path, output)

    print(f"ONNX export complete: {output}")
    print(f"Input contract: batch=1, RGB, {args.image_size}x{args.image_size}")
    print("Output contract: YOLO raw predictions; NMS is performed by our runtime")


if __name__ == "__main__":
    main()
