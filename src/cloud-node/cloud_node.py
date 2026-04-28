"""Cloud node: serves dummy video segments at multiple bitrates."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import re

# Bitrate ladder (kbps) — mimicking real ABR ladder
BITRATES_KBPS = [300, 750, 1200, 1850, 2850, 4300]
SEGMENT_DURATION_S = 4

def segment_size_bytes(bitrate_kbps):
    return bitrate_kbps * 1000 * SEGMENT_DURATION_S // 8

class SegmentHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # /segment/{bitrate_idx}/{seg_idx}
        m = re.match(r'/segment/(\d+)/(\d+)', self.path)
        if not m:
            self.send_response(404); self.end_headers(); return

        br_idx = int(m.group(1))
        if br_idx >= len(BITRATES_KBPS):
            self.send_response(400); self.end_headers(); return

        size = segment_size_bytes(BITRATES_KBPS[br_idx])
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.send_header('Content-Length', str(size))
        self.end_headers()
        # write dalam chunks biar tc bisa proper rate-limit
        chunk = b'\0' * 65536
        remaining = size
        while remaining > 0:
            n = min(remaining, len(chunk))
            self.wfile.write(chunk[:n])
            remaining -= n

    def log_message(self, *a, **kw):
        return  # silence default logs

if __name__ == "__main__":
    print("[cloud-node] serving on :8000")
    HTTPServer(('0.0.0.0', 8000), SegmentHandler).serve_forever()