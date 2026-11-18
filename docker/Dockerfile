FROM ghcr.io/nvidia/deepstream:9.2-triton-multiarch

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

WORKDIR /workspace

# DeepStream 9.2 uses CUDA 13.2 / TensorRT 10.16.0.72. The image already
# provides the native runtime; add the matching Python bindings and ONNX parser
# runtime so our Python worker can deserialize and build TensorRT engines.
ARG TENSORRT_DEB_VERSION=10.16.0.72-1+cuda13.2

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3-libnvinfer=${TENSORRT_DEB_VERSION} \
        libnvonnxparsers10=${TENSORRT_DEB_VERSION} \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install \
    --no-cache-dir \
    --break-system-packages \
    -r /tmp/requirements.txt

# Fail at image build time rather than discovering missing TensorRT bindings
# after a video has been decoded.
RUN python3 -c "import tensorrt as trt; print('TensorRT Python:', trt.__version__); assert trt.__version__.startswith('10.16.')"

# Bypass the base image's default Triton startup.
# We only need the DeepStream development environment.
ENTRYPOINT []

CMD ["bash"]
