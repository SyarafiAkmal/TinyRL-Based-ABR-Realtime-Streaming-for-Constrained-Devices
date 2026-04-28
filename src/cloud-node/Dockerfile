FROM python:3.9-slim

# iproute2 buat tc, jaringan tools buat debug
RUN apt-get update && apt-get install -y \
    iproute2 iputils-ping \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY *.py .
EXPOSE 8000
CMD ["python", "cloud_node.py"]