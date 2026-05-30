FROM python:3.9-slim

# procps for cgroup CPU sampling, build tools for av/aiortc C extensions
RUN apt-get update && apt-get install -y \
    procps \
    gcc \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    psutil \
    numpy \
    aiohttp \
    aiortc \
    av

# Shared folder untuk komunikasi dengan host/orchestrator
RUN mkdir -p /app/shared /app/video /app/model

COPY *.py .

CMD ["python", "edge_node.py", "--video", "/app/video/input.mp4"]