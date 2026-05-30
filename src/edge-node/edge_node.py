"""
edge_node.py  (upstream sender)
--------------------------------
1. Reads a local video file using PyAV via a custom VideoStreamTrack.
2. Simulates encoding delay based on bitrate + HW state (thermal/battery).
3. Every DECISION_INTERVAL seconds, calls ABR_API.get_next_bitrate()
   and applies the decision by patching aiortc's VP8 encoder bitrate.
4. Establishes a WebRTC peer connection using HTTP signaling.
5. Optionally accepts a --manual-bitrate flag to bypass the RL model.
6. Logs every ABR decision to /app/shared/abr_log.csv.

Usage:
    python edge_node.py --video /app/video/input.mp4
    python edge_node.py --video /app/video/input.mp4 --manual-bitrate 1200
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import time
import threading
from datetime import datetime
from typing import Optional

import av
import aiohttp
import aiortc.codecs.vpx as vpx_module
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, VideoStreamTrack
from av import VideoFrame

from hw_net_api import HW_Net_API
from abr_api import ABR_API, RewardConfig, BITRATE_LEVELS, _MAX_BITRATE_KBPS


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SIGNAL_URL        = "http://signaling_server:8080"
HW_STATUS_PATH    = "/app/shared/status.json"
CLOUD_NODE        = "cloud_node_container:8000"
MODEL_PATH        = "/app/model/model.npz"
LOG_PATH          = "/app/shared/abr_log.csv"
DECISION_INTERVAL = 1.0
POLL_INTERVAL     = 0.5
POLL_TIMEOUT      = 30.0
WARMUP_S          = 10.0


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------

def _init_csv_log() -> tuple:
    file_exists = os.path.exists(LOG_PATH)
    f = open(LOG_PATH, "a", newline="")
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow([
            "timestamp", "pass", "scenario",
            "lvl", "br_kbps", "conf",
            "reward", "avg20", "util",
            "thr_mbps", "loss_pct", "rtt_ms",
            "cpu_pct", "mem_pct", "thermal_c", "battery",
            "encoding_delay_ms"
        ])
        f.flush()
    return f, writer


# ---------------------------------------------------------------------------
# Bitrate-aware VideoStreamTrack with simulated encoding delay
# ---------------------------------------------------------------------------

class FileVideoTrack(VideoStreamTrack):
    """
    Reads frames from a video file (looping) and streams them.

    Simulates RPi4 encoding latency based on:
      - chosen bitrate  (higher = more work per frame)
      - thermal state   (hotter = slower CPU = longer encode)
      - battery level   (low battery = throttled CPU)

    This closes the simulation gap: the WebRTC receiver observes higher
    effective latency when the edge node is thermally throttled and
    choosing a high bitrate, matching real hardware behaviour.
    """

    def __init__(
        self,
        video_path:     str,
        bitrate_kbps:   float = 1200.0,
        hw_status_path: str   = HW_STATUS_PATH,
    ):
        super().__init__()
        self._path           = video_path
        self._container      = av.open(video_path)
        self._stream         = self._container.streams.video[0]
        self._frames         = self._container.decode(video=0)
        self._bitrate_kbps   = bitrate_kbps
        self._lock           = threading.Lock()
        self._hw_status_path = hw_status_path

    # ------------------------------------------------------------------ #

    def set_bitrate_kbps(self, kbps: float) -> None:
        with self._lock:
            self._bitrate_kbps = kbps
            vpx_module.DEFAULT_BITRATE = int(kbps * 1000)

    def get_bitrate_kbps(self) -> float:
        with self._lock:
            return self._bitrate_kbps

    # ------------------------------------------------------------------ #

    def _read_hw_status(self) -> dict:
        try:
            with open(self._hw_status_path) as f:
                return json.load(f)
        except Exception:
            return {"temp_celsius": 35.0, "battery_level": 1.0}

    def _encoding_delay_s(self, bitrate_kbps: float, hw: dict) -> float:
        """
        Simulate per-frame encoding latency on a constrained device.

        Base encoding time (RPi4 at full speed):
          150 kbps  →  5ms
          4800 kbps → 40ms   (linear interpolation)

        Thermal slowdown (beyond 70°C encoding degrades):
          ≤70°C  → 1.0x
          88°C   → ~2.5x
          ≥95°C  → ~4.0x+

        Battery slowdown (below 20% device throttles):
          >20%  → 1.0x
          ~0%   → 1.5x
        """
        base_ms = 5.0 + (bitrate_kbps / _MAX_BITRATE_KBPS) * 35.0

        temp    = hw.get("temp_celsius",  35.0)
        battery = hw.get("battery_level",  1.0)

        if temp < 70.0:
            thermal_factor = 1.0
        elif temp < 85.0:
            thermal_factor = 1.0 + (temp - 70.0) / 15.0 * 1.5
        else:
            thermal_factor = 2.5 + (temp - 85.0) / 10.0 * 1.5

        battery_factor = (
            1.0 if battery > 0.2
            else 1.0 + (0.2 - battery) / 0.2 * 0.5
        )

        return (base_ms * thermal_factor * battery_factor) / 1000.0

    # ------------------------------------------------------------------ #

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()

        try:
            frame = next(self._frames)
        except StopIteration:
            self._container.close()
            self._container = av.open(self._path)
            self._stream    = self._container.streams.video[0]
            self._frames    = self._container.decode(video=0)
            frame           = next(self._frames)

        # Simulate encoding delay before handing frame to WebRTC
        with self._lock:
            current_kbps = self._bitrate_kbps

        hw    = self._read_hw_status()
        delay = self._encoding_delay_s(current_kbps, hw)
        await asyncio.sleep(delay)

        video_frame         = frame.to_ndarray(format="bgr24")
        new_frame           = VideoFrame.from_ndarray(video_frame, format="bgr24")
        new_frame.pts       = pts
        new_frame.time_base = time_base

        with self._lock:
            vpx_module.DEFAULT_BITRATE = int(self._bitrate_kbps * 1000)

        return new_frame

    def close(self) -> None:
        self._container.close()


# ---------------------------------------------------------------------------
# Signaling helpers
# ---------------------------------------------------------------------------

async def post_offer(session: aiohttp.ClientSession, sdp: str, sdp_type: str) -> None:
    await session.post(f"{SIGNAL_URL}/offer", json={"sdp": sdp, "type": sdp_type})


async def poll_answer(session: aiohttp.ClientSession) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT
    while time.monotonic() < deadline:
        async with session.get(f"{SIGNAL_URL}/answer") as resp:
            if resp.status == 200:
                return await resp.json()
        await asyncio.sleep(POLL_INTERVAL)
    raise TimeoutError("Timed out waiting for SDP answer from cloud node")


# ---------------------------------------------------------------------------
# Shared status reader
# ---------------------------------------------------------------------------

def _read_scenario_info() -> tuple[str, int]:
    try:
        with open(HW_STATUS_PATH) as f:
            status = json.load(f)
        return status.get("scenario", "unknown"), status.get("pass", 0)
    except Exception:
        return "unknown", 0


# ---------------------------------------------------------------------------
# ABR decision loop
# ---------------------------------------------------------------------------

async def abr_loop(
    abr_api:     ABR_API,
    track:       FileVideoTrack,
    manual_kbps: Optional[float],
    stop_event:  asyncio.Event,
) -> None:
    """
    Background task. Every DECISION_INTERVAL seconds:
      - Manual mode : apply fixed kbps, no model.
      - RL mode     : observe state → infer → apply bitrate →
                      compute reward (QoE + utilization) → update PPO.

    Encoding delay is injected in FileVideoTrack.recv() so the network
    probe's observed RTT already reflects HW stress + bitrate choice.
    """
    from abr_api import HWState

    reward_history: list[float] = []
    scenario    = "unknown"
    pass_count  = 0

    csv_file, csv_writer = _init_csv_log()

    print(f"[abr] warming up probes for {WARMUP_S:.0f}s...", flush=True)
    await asyncio.sleep(WARMUP_S)
    print("[abr] starting decisions", flush=True)

    while not stop_event.is_set():
        await asyncio.sleep(DECISION_INTERVAL)

        scenario, pass_count = _read_scenario_info()

        # ── Manual mode ──────────────────────────────────────────────────
        if manual_kbps is not None:
            track.set_bitrate_kbps(manual_kbps)
            print(f"[abr] MANUAL  bitrate={manual_kbps:.0f}kbps", flush=True)
            continue

        # ── RL model mode ─────────────────────────────────────────────────
        try:
            decision = abr_api.get_next_bitrate()
        except Exception as e:
            print(f"[abr] model error: {e}", flush=True)
            continue

        track.set_bitrate_kbps(decision.bitrate_kbps)

        # Read HW status for encoding delay logging
        try:
            with open(HW_STATUS_PATH) as f:
                hw_status = json.load(f)
        except Exception:
            hw_status = {"temp_celsius": 35.0, "battery_level": 1.0}

        encoding_delay_ms = track._encoding_delay_s(
            decision.bitrate_kbps, hw_status
        ) * 1000.0

        net          = abr_api.hw_net_api.get_net_state()
        hw_state_raw = abr_api.hw_net_api.get_hw_state()

        # Skip unstable probes
        if net["estimated_throughput"] > 100.0 or net["rtt_ms"] < 5.0:
            print(
                f"[abr] skipping — probe not stable "
                f"(thr={net['estimated_throughput']:.1f} rtt={net['rtt_ms']:.1f})",
                flush=True,
            )
            if abr_api._agent._buf_actions:
                abr_api._agent._buf_actions.pop()
                abr_api._agent._buf_states.pop()
                abr_api._agent._buf_rewards.pop()
                abr_api._agent._buf_log_probs.pop()
                abr_api._agent._buf_values.pop()
                abr_api._agent._buf_dones.pop()
            continue

        hw_snap = HWState(
            cpu_pressure    = hw_state_raw["cpu_pressure"],
            memory_pressure = hw_state_raw["memory_pressure"],
            thermal_state   = hw_state_raw["thermal_state"],
            battery_level   = hw_state_raw["battery_level"],
        )

        switched_to_gcc = net["packet_loss_rate"] > 0.15

        # ── Reward ────────────────────────────────────────────────────────
        reward = abr_api.reward_calc.compute(
            throughput_mbps  = net["estimated_throughput"],
            packet_loss_rate = net["packet_loss_rate"],
            rtt_ms           = net["rtt_ms"],
            switched_to_gcc  = switched_to_gcc,
            hw               = hw_snap,
            chosen_kbps      = decision.bitrate_kbps,
        )

        # ── Utilization reward ────────────────────────────────────────────
        available_kbps = net["estimated_throughput"] * 1000.0
        if available_kbps > 0:
            utilization        = decision.bitrate_kbps / available_kbps
            utilization_reward = 1.0 - abs(utilization - 0.75) * 2.0
            utilization_reward = max(-1.0, min(1.0, utilization_reward))
        else:
            utilization        = 0.0
            utilization_reward = -1.0

        reward += 0.5 * utilization_reward

        # ── Rolling average ───────────────────────────────────────────────
        reward_history.append(reward)
        avg20 = sum(reward_history[-20:]) / min(len(reward_history), 20)

        abr_api.update(reward)

        # ── CSV log ───────────────────────────────────────────────────────
        csv_writer.writerow([
            datetime.utcnow().isoformat(),
            pass_count,
            scenario,
            decision.bitrate_level,
            decision.bitrate_kbps,
            round(decision.confidence, 4),
            round(reward, 4),
            round(avg20, 4),
            round(utilization, 3),
            round(net["estimated_throughput"], 2),
            round(net["packet_loss_rate"] * 100, 2),
            round(net["rtt_ms"], 1),
            round(hw_state_raw["cpu_pressure"], 1),
            round(hw_state_raw["memory_pressure"], 1),
            round(hw_state_raw["thermal_state"], 1),
            round(hw_state_raw["battery_level"], 3),
            round(encoding_delay_ms, 2),
        ])
        csv_file.flush()

        # ── Console log ───────────────────────────────────────────────────
        print(
            f"[abr] RL  pass={pass_count}  sc={scenario}  "
            f"lvl={decision.bitrate_level}  br={decision.bitrate_kbps:.0f}kbps  "
            f"conf={decision.confidence:.3f}  reward={reward:.4f}  avg20={avg20:.4f}  "
            f"util={utilization:.2f}  thr={net['estimated_throughput']:.2f}Mbps  "
            f"loss={net['packet_loss_rate']*100:.1f}%  rtt={net['rtt_ms']:.0f}ms  "
            f"enc={encoding_delay_ms:.1f}ms",
            flush=True,
        )

    csv_file.close()

    try:
        abr_api.save_model()
        print(f"[abr] model saved → {MODEL_PATH}", flush=True)
    except Exception as e:
        print(f"[abr] model save failed: {e}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run(args: argparse.Namespace) -> None:
    manual_kbps: Optional[float] = args.manual_bitrate

    hw_net_api = HW_Net_API(
        shared_status_path = HW_STATUS_PATH,
        target_node        = CLOUD_NODE,
    )

    reward_cfg = RewardConfig(
        alpha   = 2.0,
        beta    = 3.0,
        gamma   = 0.5,
        delta   = 1.5,
        epsilon = 0.1,
        zeta    = 1.0,
    )
    abr_api = ABR_API(MODEL_PATH, hw_net_api, reward_config=reward_cfg)

    init_kbps = manual_kbps if manual_kbps else BITRATE_LEVELS[2]
    track = FileVideoTrack(args.video, bitrate_kbps=init_kbps)
    print(f"[edge] video file: {args.video}  init bitrate: {init_kbps}kbps", flush=True)

    pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    pc.addTrack(track)

    @pc.on("connectionstatechange")
    async def on_state():
        print(f"[edge] connection state → {pc.connectionState}", flush=True)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    async with aiohttp.ClientSession() as session:
        await post_offer(session, pc.localDescription.sdp, pc.localDescription.type)
        print("[edge] offer posted, waiting for answer...", flush=True)
        answer_data = await poll_answer(session)

    answer = RTCSessionDescription(sdp=answer_data["sdp"], type=answer_data["type"])
    await pc.setRemoteDescription(answer)
    print("[edge] WebRTC connected", flush=True)

    stop_event = asyncio.Event()
    abr_task   = asyncio.create_task(
        abr_loop(abr_api, track, manual_kbps, stop_event)
    )

    try:
        while pc.connectionState not in ("failed", "closed"):
            await asyncio.sleep(1.0)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stop_event.set()
        await abr_task
        await pc.close()
        track.close()
        print("[edge] shutdown complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Edge node WebRTC upstream sender")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument(
        "--manual-bitrate", type=float, default=None, metavar="KBPS",
        help="Bypass RL model and force a fixed bitrate in kbps",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()