FROM python:3.9-slim

# iproute2 for tc/network shaping, build tools for av/aiortc C extensions
RUN apt-get update && apt-get install -y \
    iproute2 \
    iputils-ping \
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
    aiohttp \
    aiortc \
    av

RUN mkdir -p /app/shared

COPY *.py .

EXPOSE 8000
CMD ["python", "cloud_node.py"]