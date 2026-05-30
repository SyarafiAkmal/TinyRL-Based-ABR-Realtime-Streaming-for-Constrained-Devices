FROM python:3.11-slim
WORKDIR /app
RUN pip install aiohttp
COPY signaling_server.py .
CMD ["python", "signaling_server.py"]