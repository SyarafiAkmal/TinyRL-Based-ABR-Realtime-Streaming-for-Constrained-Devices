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
                self.status = {
                    "temp_celsius": 35,
                    "battery_level": 1.0,
                    "rate": "10mbit",
                    "delay": "20ms",
                    "loss": "0.1%"
                }

        self.target_node = target_node

        self._throughput_window: deque = deque(maxlen=5)
        self._probe_results:     deque = deque(maxlen=10)
        self._previous_bitrate_kbps: float = 0.0

    # =========================================================================
    # Public API
    # =========================================================================

    def refresh_state(self):
        """Reload the shared metrics JSON to get updated simulated values."""
        with open("/app/shared/status.json", "r") as f:
            try:
                self.status = json.load(f)
            except:
                self.status = {
                    "temp_celsius": 35,
                    "battery_level": 1.0,
                    "rate": "10mbit",
                    "delay": "20ms",
                    "loss": "0.1%"
                }

    def set_previous_bitrate(self, bitrate_kbps: float):
        self._previous_bitrate_kbps = bitrate_kbps

    # =========================================================================
    # Hardware API
    # =========================================================================

    def get_hw_state(self) -> dict:
        """
        Returns real container CPU/memory alongside simulated thermal
        and battery values from the shared metrics file.

        CPU pressure combines real cgroup measurement with a thermal
        throttle component — matching how a real RPi4 reduces clock
        speed (and therefore effective CPU capacity) when overheating.

        Mapping:
          temp_celsius  → thermal throttle extra CPU pressure
                          (simulates RPi4 reducing clock at high temp)
          battery_level → already feeds into encoding_delay in FileVideoTrack
                          and into C_limit in the orchestrator

        Returns:
            {
                "cpu_pressure":     float   # % (0-100), real + thermal overhead
                "memory_pressure":  float   # % (0-100), real cgroup
                "thermal_state":    float   # degrees Celsius, injected
                "battery_level":    float   # fraction 0.0-1.0, injected
            }
        """
        self.refresh_state()

        real_cpu = self.get_cpu_pressure()
        temp     = self.status.get("temp_celsius", 35.0)

        # Thermal throttle component:
        # Beyond 70°C a real RPi4 starts reducing clock speed.
        # We model this as additional CPU pressure on top of real cgroup usage:
        #   70°C → +0%   (no throttle yet)
        #   85°C → +18%  (moderate throttle)
        #   95°C → +30%  (heavy throttle, capped)
        if temp > 70.0:
            thermal_extra = min(((temp - 70.0) / 25.0) * 30.0, 30.0)
        else:
            thermal_extra = 0.0

        simulated_cpu = min(100.0, real_cpu + thermal_extra)

        return {
            "cpu_pressure":    simulated_cpu,
            "memory_pressure": self.get_memory_pressure(),
            "thermal_state":   temp,
            "battery_level":   self.status.get("battery_level", 1.0),
        }

    # =========================================================================
    # Hardware Metrics — CPU
    # =========================================================================

    def _read_cpu_usage_usec(self) -> int:
        with open("/sys/fs/cgroup/cpu.stat") as f:
            for line in f:
                if line.startswith("usage_usec"):
                    return int(line.split()[1])
        return 0

    def _read_cpu_cap(self) -> float:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().strip().split()
        if parts[0] == "max":
            return float(os.cpu_count() or 1)
        quota_us, period_us = int(parts[0]), int(parts[1])
        return quota_us / period_us

    def get_cpu_pressure(self, sample_window_s: float = 0.1) -> float:
        """
        Sample CPU usage over a small window, return % of cap consumed.
        0   = idle
        100 = saturating cap
        """
        cap = self._read_cpu_cap()
        if cap <= 0:
            return 0.0

        t0     = self._read_cpu_usage_usec()
        wall_t0 = time.time()
        time.sleep(sample_window_s)
        t1           = self._read_cpu_usage_usec()
        wall_elapsed = time.time() - wall_t0

        cpu_used_s = (t1 - t0) / 1e6
        cpu_abs    = (cpu_used_s / wall_elapsed) * 100 if wall_elapsed > 0 else 0.0
        return min(100.0, cpu_abs / cap)

    # =========================================================================
    # Hardware Metrics — Memory
    # =========================================================================

    def get_memory_pressure(self) -> float:
        """% of memory limit currently in use."""
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
        """
        Probe target_node and return current upstream network observation.

        Returns:
            {
                "estimated_throughput":  float   # Mbps
                "rtt_ms":                float   # ms
                "packet_loss_rate":      float   # 0.0–1.0
                "previous_bitrate_kbps": float
            }
        """
        rtt_ms, throughput = self._probe_network()
        return {
            "estimated_throughput":  throughput,
            "rtt_ms":                rtt_ms,
            "packet_loss_rate":      self._estimated_loss_rate(),
            "previous_bitrate_kbps": self._previous_bitrate_kbps,
        }

    def _probe_network(self) -> tuple[float, float]:
        rtt_ms        = self._probe_rtt()
        throughput_mbps = self._probe_throughput()
        return rtt_ms, throughput_mbps

    def _probe_rtt(self) -> float:
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
        url = f"http://{self.target_node}/probe"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                t_first_byte = time.perf_counter()
                data         = resp.read()
                elapsed      = time.perf_counter() - t_first_byte
            if elapsed <= 0:
                self._probe_results.append(False)
                return 0.0
            self._probe_results.append(True)
            mbps = (len(data) * 8 / 1e6) / elapsed
            self._throughput_window.append(mbps)
            return self._smoothed_throughput()
        except (urllib.error.URLError, OSError):
            self._probe_results.append(False)
            return 0.0

    def _smoothed_throughput(self) -> float:
        if not self._throughput_window:
            return 0.0
        return sum(self._throughput_window) / len(self._throughput_window)

    def _estimated_loss_rate(self) -> float:
        if not self._probe_results:
            return 0.0
        failures = sum(1 for s in self._probe_results if not s)
        return failures / len(self._probe_results)