FROM ghcr.io/nvidia/deepstream:9.2-triton-multiarch

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN python3 -m pip install \
    --no-cache-dir \
    --break-system-packages \
    -r /tmp/requirements.txt

# Bypass the base image's default Triton startup.
# We only need the DeepStream development environment.
ENTRYPOINT []

CMD ["bash"]
