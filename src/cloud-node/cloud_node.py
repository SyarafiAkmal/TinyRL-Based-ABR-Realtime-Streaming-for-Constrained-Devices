"""Cloud node: serves dummy video segments + probe endpoint."""
from http.server import BaseHTTPRequestHandler, HTTPServer

# Bitrate ladder (kbps)
BITRATES_KBPS = [300, 750, 1200, 1850, 2850, 4300]
SEGMENT_DURATION_S = 4
PROBE_SIZE_BYTES = 100_000  # 100KB for throughput estimation


def segment_size(bitrate_kbps: int) -> int:
    """Return segment size in bytes for given bitrate."""
    return bitrate_kbps * 1000 * SEGMENT_DURATION_S // 8


def serve_bytes(handler: BaseHTTPRequestHandler, n_bytes: int):
    """Write n_bytes of zero-filled payload. Used for both /probe and /segment."""
    handler.send_response(200)
    handler.send_header('Content-Type', 'application/octet-stream')
    handler.send_header('Content-Length', str(n_bytes))
    handler.end_headers()
    chunk = b'\0' * 65536
    remaining = n_bytes
    while remaining > 0:
        handler.wfile.write(chunk[:min(remaining, len(chunk))])
        remaining -= min(remaining, len(chunk))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path

        # /probe → 100KB throughput test
        if path == '/probe':
            serve_bytes(self, PROBE_SIZE_BYTES)
            return

        # /segment/{bitrate_idx} → dummy video segment
        if path.startswith('/segment/'):
            try:
                br_idx = int(path.split('/')[2])
                if 0 <= br_idx < len(BITRATES_KBPS):
                    serve_bytes(self, segment_size(BITRATES_KBPS[br_idx]))
                    return
            except (ValueError, IndexError):
                pass

        self.send_response(404)
        self.end_headers()

    def log_message(self, *args, **kwargs):
        pass  # silence default access logs


if __name__ == "__main__":
    print(f"[cloud-node] serving on :8000  bitrates={BITRATES_KBPS} kbps")
    HTTPServer(('0.0.0.0', 8000), Handler).serve_forever()