import math
import os
import time
import json
import threading
import urllib.request
from collections import deque
from typing import Optional


class HW_Net_API:
    def __init__(self, shared_status_path: str, target_node: str):
        """
        Args:
            shared_status_path: Path to JSON file with simulated HW states
                                 (temp_celsius, battery_level, etc.)
            target_node:         Hostname or IP of the streaming/CDN node
                                 used for network probing.
        """

        with open(shared_status_path, "r") as f:
            try:
                self.status = json.load(f)
            except:
                self.status = {  # optimal conditions
                    "temp_celsius": 35,
                    "battery_level": 1.0,
                    "rate": "10mbit",
                    "delay": "20ms",
                    "loss": "0.1%"
                }

        self.target_node = target_node

        # Network probe state
        self._throughput_window: deque = deque(maxlen=5)   # expanded to 5 for upstream stability
        self._probe_results: deque = deque(maxlen=10)      # True=success, False=failure (for loss rate)
        self._previous_bitrate_kbps: float = 0.0

    # =========================================================================
    # Public API
    # =========================================================================

    def refresh_state(self):
        """Reload the shared metrics JSON to get updated simulated values from the orchestrator."""
        with open("/app/shared/status.json", "r") as f:
            try:
                self.status = json.load(f)
            except:
                self.status = {  # optimal conditions
                    "temp_celsius": 35,
                    "battery_level": 1.0,
                    "rate": "10mbit",
                    "delay": "20ms",
                    "loss": "0.1%"
                }

    def set_previous_bitrate(self, bitrate_kbps: float):
        """
        Call this whenever the encoder commits to a new output bitrate.

        Args:
            bitrate_kbps: The bitrate (kbps) the encoder is currently targeting.
        """
        self._previous_bitrate_kbps = bitrate_kbps

    # =========================================================================
    # Hardware API
    # =========================================================================

    def get_hw_state(self) -> dict:
        """
        Returns real container CPU/memory alongside simulated
        thermal and battery values from the shared metrics file.

        Returns:
            {
                "cpu_pressure":     float   # % (0-100)
                "memory_pressure":  float   # % (0-100)
                "thermal_state":    float   # degrees Celsius
                "battery_level":    float   # fraction (0.0-1.0)
            }
        """
        self.refresh_state()
        return {
            "cpu_pressure":    self.get_cpu_pressure(),
            "memory_pressure": self.get_memory_pressure(),
            "thermal_state":   self.status.get("temp_celsius"),
            "battery_level":   self.status.get("battery_level"),
        }

    # =========================================================================
    # Hardware Metrics — CPU
    # =========================================================================

    def _read_cpu_usage_usec(self) -> int:
        """Cumulative CPU time consumed by container (microseconds)."""
        with open("/sys/fs/cgroup/cpu.stat") as f:
            for line in f:
                if line.startswith("usage_usec"):
                    return int(line.split()[1])
        return 0

    def _read_cpu_cap(self) -> float:
        """Active CPU cap (fraction of one core). Returns N cores if uncapped."""
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().strip().split()
        if parts[0] == "max":
            return float(os.cpu_count() or 1)
        quota_us, period_us = int(parts[0]), int(parts[1])
        return quota_us / period_us

    def get_cpu_pressure(self, sample_window_s: float = 0.1) -> float:
        """Sample CPU usage over a small window, return % of cap consumed.

        0   = idle
        100 = saturating cap (cannot use more)
        """
        cap = self._read_cpu_cap()
        if cap <= 0:
            return 0.0

        t0 = self._read_cpu_usage_usec()
        wall_t0 = time.time()
        time.sleep(sample_window_s)
        t1 = self._read_cpu_usage_usec()
        wall_elapsed = time.time() - wall_t0

        cpu_used_s = (t1 - t0) / 1e6
        cpu_abs = (cpu_used_s / wall_elapsed) * 100 if wall_elapsed > 0 else 0.0
        return min(100.0, cpu_abs / cap)

    # =========================================================================
    # Hardware Metrics — Memory
    # =========================================================================

    def get_memory_pressure(self) -> float:
        """% of memory limit currently in use. Instantaneous read, no sampling needed."""
        with open("/sys/fs/cgroup/memory.current") as f:
            usage = int(f.read().strip())
        with open("/sys/fs/cgroup/memory.max") as f:
            raw = f.read().strip()

        if raw == "max":
            limit = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        else:
            limit = int(raw)

        return (usage / limit) * 100 if limit > 0 else 0.0

    # =========================================================================
    # Network API
    # =========================================================================

    def get_net_state(self) -> dict:
        """Probe target_node and return current upstream network observation.

        Returns:
            {
                "estimated_throughput":  float   # Mbps, smoothed over last 5 probes
                "rtt_ms":                float   # round-trip time in milliseconds
                "packet_loss_rate":      float   # fraction 0.0–1.0, estimated over last 10 probes
                "previous_bitrate_kbps": float   # last bitrate set via set_previous_bitrate()
            }

        Note: Each call triggers an HTTP probe (~tens of ms to seconds depending
        on network shaping). Prefer calling sparsely.
        """
        rtt_ms, throughput = self._probe_network()
        return {
            "estimated_throughput":  throughput,
            "rtt_ms":                rtt_ms,
            "packet_loss_rate":      self._estimated_loss_rate(),
            "previous_bitrate_kbps": self._previous_bitrate_kbps,
        }

    def _probe_network(self) -> tuple[float, float]:
        rtt_ms = self._probe_rtt()
        throughput_mbps = self._probe_throughput()
        return rtt_ms, throughput_mbps

    def _probe_rtt(self) -> float:
        """Small request → approximates RTT (still includes TCP handshake)."""
        url = f"http://{self.target_node}/ping"
        try:
            t0 = time.perf_counter()
            with urllib.request.urlopen(url, timeout=2) as resp:
                resp.read()
            return (time.perf_counter() - t0) * 1000.0
        except (urllib.error.URLError, OSError):
            self._probe_results.append(False)
            return 0.0

    def _probe_throughput(self) -> float:
        """Large payload → measure goodput from first-byte to last-byte."""
        url = f"http://{self.target_node}/probe"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                t_first_byte = time.perf_counter()        # start AFTER headers
                data = resp.read()
                elapsed = time.perf_counter() - t_first_byte
            if elapsed <= 0:
                self._probe_results.append(False); return 0.0
            self._probe_results.append(True)
            mbps = (len(data) * 8 / 1e6) / elapsed
            self._throughput_window.append(mbps)
            return self._smoothed_throughput()
        except (urllib.error.URLError, OSError):
            self._probe_results.append(False); return 0.0

    def _smoothed_throughput(self) -> float:
        """Sliding-window average throughput (Mbps) over last 5 probes."""
        if not self._throughput_window:
            return 0.0
        return sum(self._throughput_window) / len(self._throughput_window)

    def _estimated_loss_rate(self) -> float:
        """Fraction of recent probes that failed, over a rolling window of 10.

        Returns 0.0 if no probes have been attempted yet.
        """
        if not self._probe_results:
            return 0.0
        failures = sum(1 for success in self._probe_results if not success)
        return failures / len(self._probe_results)