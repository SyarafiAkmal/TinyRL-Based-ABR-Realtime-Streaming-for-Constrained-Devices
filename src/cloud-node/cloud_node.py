"""
cloud_node.py  (upstream receiver)
------------------------------------
Combines two servers in one process:

1. HTTP server  (:8000)
   - /ping       → 0-byte RTT probe
   - /probe      → 100 KB throughput probe
   - /segment/N  → dummy video segment bytes
   - /stats      → JSON snapshot of WebRTC receive stats

2. WebRTC signaling consumer (:8080 via signaling_server)
   - Polls GET /offer from the signaling server
   - Accepts the peer connection
   - Receives the video track from the edge node
   - Logs per-frame bitrate and cumulative stats

Stats are written to /app/shared/webrtc_stats.json every 5 s so the
orchestrator or a monitoring script can read them.
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Optional

import aiohttp
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack, RTCConfiguration
from aiortc.contrib.media import MediaRecorder

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIGNAL_URL    = "http://signaling_server:8080"
HTTP_PORT     = 8000
STATS_PATH    = "/app/shared/webrtc_stats.json"
POLL_INTERVAL = 1.0    # seconds between /offer polls
STATS_INTERVAL = 5.0   # seconds between stats flush

BITRATES_KBPS      = [300, 750, 1200, 1850, 2850, 4300]
SEGMENT_DURATION_S = 4
PROBE_SIZE_BYTES   = 100_000


# ---------------------------------------------------------------------------
# Shared stats (written by WebRTC receiver, read by /stats endpoint)
# ---------------------------------------------------------------------------

_stats: dict = {
    "frames_received":   0,
    "bytes_received":    0,
    "estimated_kbps":    0.0,
    "last_frame_ts":     0.0,
    "connection_state":  "new",
}
_stats_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# HTTP server (sync, runs in a thread)
# ---------------------------------------------------------------------------

def _segment_size(bitrate_kbps: int) -> int:
    return bitrate_kbps * 1000 * SEGMENT_DURATION_S // 8


def _serve_bytes(handler: BaseHTTPRequestHandler, n_bytes: int) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", "application/octet-stream")
    handler.send_header("Content-Length", str(n_bytes))
    handler.end_headers()
    chunk     = b"\0" * 65536
    remaining = n_bytes
    while remaining > 0:
        to_write = min(remaining, len(chunk))
        handler.wfile.write(chunk[:to_write])
        remaining -= to_write


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path

        if path == "/ping":
            _serve_bytes(self, 0)
            return

        if path == "/probe":
            _serve_bytes(self, PROBE_SIZE_BYTES)
            return

        if path == "/stats":
            body = json.dumps(_stats).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path.startswith("/segment/"):
            try:
                idx = int(path.split("/")[2])
                if 0 <= idx < len(BITRATES_KBPS):
                    _serve_bytes(self, _segment_size(BITRATES_KBPS[idx]))
                    return
            except (ValueError, IndexError):
                pass

        self.send_response(404)
        self.end_headers()

    def log_message(self, *args, **kwargs):
        pass  # silence default HTTP logs


def _run_http_server() -> None:
    print(f"[cloud] HTTP server on :{HTTP_PORT}", flush=True)
    HTTPServer(("0.0.0.0", HTTP_PORT), _Handler).serve_forever()


# ---------------------------------------------------------------------------
# WebRTC receiver
# ---------------------------------------------------------------------------

class _StatsTrack(MediaStreamTrack):
    """
    Wraps the incoming video track to collect per-frame stats.
    Does not re-encode — purely observational.
    """
    kind = "video"

    def __init__(self, track: MediaStreamTrack):
        super().__init__()
        self._track          = track
        self._frame_count    = 0
        self._byte_count     = 0
        self._window_start   = time.monotonic()
        self._window_bytes   = 0
        self._window_frames  = 0
        self._est_kbps       = 0.0

    async def recv(self):
        frame = await self._track.recv()

        # Rough byte estimate: width * height * 3 / compression_factor
        # VP8 typically achieves ~30:1 for video content
        raw_bytes  = frame.width * frame.height * 3
        comp_bytes = raw_bytes // 30

        now = time.monotonic()
        self._frame_count   += 1
        self._byte_count    += comp_bytes
        self._window_bytes  += comp_bytes
        self._window_frames += 1

        # Recalculate estimated kbps every second
        elapsed = now - self._window_start
        if elapsed >= 1.0:
            self._est_kbps      = (self._window_bytes * 8) / elapsed / 1000.0
            self._window_bytes  = 0
            self._window_frames = 0
            self._window_start  = now

        # Update global stats (non-blocking best-effort)
        _stats["frames_received"] = self._frame_count
        _stats["bytes_received"]  = self._byte_count
        _stats["estimated_kbps"]  = round(self._est_kbps, 1)
        _stats["last_frame_ts"]   = now

        return frame


async def _flush_stats_loop() -> None:
    """Periodically persist stats to disk."""
    stats_path = Path(STATS_PATH)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        await asyncio.sleep(STATS_INTERVAL)
        try:
            stats_path.write_text(json.dumps(_stats, indent=2))
        except Exception as e:
            print(f"[cloud] stats flush error: {e}", flush=True)


async def _webrtc_receiver_loop() -> None:
    """
    Polls for SDP offer, accepts it, handles the track.
    Loops forever — restarts the session if connection drops.
    """
    while True:
        async with aiohttp.ClientSession() as session:

            # Reset previous session on signaling server before polling
            try:
                await session.delete(f"{SIGNAL_URL}/session")
            except Exception:
                pass

            print("[cloud] waiting for SDP offer...", flush=True)

            while True:
                try:
                    async with session.get(f"{SIGNAL_URL}/offer") as resp:
                        if resp.status == 200:
                            offer_data = await resp.json()
                            break
                except Exception:
                    pass
                await asyncio.sleep(POLL_INTERVAL)

            print("[cloud] offer received, creating peer connection", flush=True)

            pc = RTCPeerConnection(
                configuration=RTCConfiguration(iceServers=[])
            )
            _stats["connection_state"] = "connecting"

            drain_task = None

            @pc.on("connectionstatechange")
            async def on_state():
                _stats["connection_state"] = pc.connectionState
                print(f"[cloud] connection state → {pc.connectionState}", flush=True)

            @pc.on("track")
            async def on_track(track: MediaStreamTrack):
                nonlocal drain_task
                print(f"[cloud] track received: kind={track.kind}", flush=True)
                if track.kind == "video":
                    stats_track = _StatsTrack(track)
                    drain_task  = asyncio.ensure_future(_drain_track(stats_track))

            # SDP negotiation
            offer  = RTCSessionDescription(sdp=offer_data["sdp"], type=offer_data["type"])
            await pc.setRemoteDescription(offer)
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)

            await session.post(
                f"{SIGNAL_URL}/answer",
                json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
            )
            print("[cloud] answer posted, streaming...", flush=True)

            # Wait until connection drops
            while pc.connectionState not in ("failed", "closed", "disconnected"):
                await asyncio.sleep(1.0)

            print("[cloud] connection ended, restarting session...", flush=True)

            if drain_task and not drain_task.done():
                drain_task.cancel()

            await pc.close()
            await asyncio.sleep(2.0)   # brief pause before restarting


async def _drain_track(track: _StatsTrack) -> None:
    """Consume frames from the stats track indefinitely."""
    try:
        while True:
            frame = await track.recv()
            # Frame is available here for further processing if needed
            # e.g. writing to disk, running inference, etc.
    except Exception as e:
        print(f"[cloud] track drain ended: {e}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def _async_main() -> None:
    # HTTP server in a daemon thread
    http_thread = Thread(target=_run_http_server, daemon=True)
    http_thread.start()

    # Stats flush loop + WebRTC receiver concurrently
    await asyncio.gather(
        _flush_stats_loop(),
        _webrtc_receiver_loop(),
    )


if __name__ == "__main__":
    asyncio.run(_async_main())