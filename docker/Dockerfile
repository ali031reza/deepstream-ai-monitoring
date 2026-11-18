FROM ghcr.io/nvidia/deepstream:9.2-triton-multiarch

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

WORKDIR /workspace

# DeepStream 9.2 already provides the native TensorRT 10.16 runtime,
# ONNX parser library, CUDA runtime, and trtexec. Install only the matching
# Python package from NVIDIA's PyPI distribution. Avoid apt-get update here:
# the base image's CUDA APT source can be unavailable independently of PyPI.
ARG TENSORRT_PYTHON_VERSION=10.16.0.72

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install \
    --no-cache-dir \
    --break-system-packages \
    --extra-index-url https://pypi.nvidia.com \
    --only-binary=tensorrt-cu13-libs,tensorrt-cu13-bindings \
    -r /tmp/requirements.txt \
    "tensorrt-cu13==${TENSORRT_PYTHON_VERSION}"

# Fail at image build time rather than discovering missing TensorRT bindings
# after a video has been decoded.
RUN python3 - <<'PY'
import tensorrt as trt

print("TensorRT Python:", trt.__version__)
assert trt.__version__ == "10.16.0.72", trt.__version__
assert trt.Builder(trt.Logger()) is not None
assert hasattr(trt, "OnnxParser")
print("TensorRT Builder and ONNX parser bindings: OK")
PY

RUN ldconfig -p | grep -q 'libnvonnxparser.so.10' \
    && command -v trtexec

# Bypass the base image's default Triton startup.
# We only need the DeepStream development environment.
ENTRYPOINT []

CMD ["bash"]
