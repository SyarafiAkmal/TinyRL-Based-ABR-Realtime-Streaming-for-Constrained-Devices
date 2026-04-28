FROM python:3.9-slim

# Install procps supaya psutil bisa akses sistem proses
RUN apt-get update && apt-get install -y \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install library wajib untuk monitor & math
RUN pip install --no-cache-dir psutil numpy

# Buat folder shared untuk komunikasi dengan Host
RUN mkdir -p /app/shared

# Copy semua file .py yang ada di folder edge-node ke dalam container
# Ini lebih aman daripada copy satu-satu kalau filenya nambah
COPY *.py .

# Jalankan script utama kamu
CMD ["python", "edge_node.py"]