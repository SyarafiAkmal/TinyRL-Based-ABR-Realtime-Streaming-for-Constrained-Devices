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
            state:               Active state key in the metrics JSON
                                 (e.g. "normal", "low_battery", "throttled")
            target_node:         Hostname or IP of the streaming/CDN node
                                 used for network probing.
        """

        with open(shared_status_path, "r") as f:
            try:
                self.status = json.load(f)
            except:
                self.status = { # optimal conditions
                    "temp_celsius": 35,
                    "battery_level": 1.0,
                    "rate": "10mbit",
                    "delay": "20ms",
                    "loss": "0.1%"
                }

        self.target_node = target_node

        # Network probe state
        self._throughput_window: deque = deque(maxlen=2)
        self._last_fetch_time_s: float = 0.0

    # =========================================================================
    # Public API
    # =========================================================================

    def refresh_state(self):
        """Reload the shared metrics JSON to get updated simulated values from the orchestrator."""
        with open("/app/shared/status.json", "r") as f:
            try:
                self.status = json.load(f)
            except:
                self.status = { # optimal conditions
                    "temp_celsius": 35,
                    "battery_level": 1.0,
                    "rate": "10mbit",
                    "delay": "20ms",
                    "loss": "0.1%"
                }

    # =========================================================================
    # Hardware API
    # =========================================================================

    def get_hw_state(self) -> dict:
        """
        Returns real container CPU/memory alongside simulated
        thermal and battery values from the shared metrics file.

        Returns:
            {
                "cpu_pressure":  float   # % (0-100)
                "memory_pressure":  float   # % (0-100)
                "thermal_state":    float   # severity scale (0.0-1.0) scaled on 20-100°C
                "battery_level":    float   # % (0.0-1.0)
            }
        """

        self.refresh_state()
        return {
            "cpu_pressure": self.get_cpu_pressure(),
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
        """Return current network probe conditions as seen from edge node to target_node.
        
        Returns:
            {
                "segment_fetch_time": float   # e.g. "50ms"
                "estimated_throughput":  float   # e.g. "0.1%"
            }
        """
        self.refresh_state()
        pass

    def get_net_state(self) -> dict:
        """Probe target_node and return current network observation.
        
        Returns:
            {
                "segment_fetch_time":   float   # seconds for last probe
                "estimated_throughput": float   # Mbps (smoothed over last 5 probes)
            }
        
        Note: Each call triggers an HTTP probe (~tens of ms to seconds depending
        on network shaping). Prefer calling sparsely.
        """
        fetch_time, throughput = self._probe_network()
        return {
            "segment_fetch_time":   fetch_time,
            "estimated_throughput": throughput,
        }
    
    def _probe_network(self) -> tuple[float, float]:
        """HTTP probe to target_node/probe endpoint.
        
        Returns:
            (fetch_time_s, throughput_mbps)
            Returns (0.0, 0.0) on failure.
        """
        url = f"http://{self.target_node}/probe"
        try:
            t_start = time.perf_counter()
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = resp.read()
            elapsed_s = time.perf_counter() - t_start
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            return 0.0, 0.0

        if elapsed_s <= 0:
            return 0.0, 0.0

        size_bits = len(data) * 8
        throughput_mbps = (size_bits / 1e6) / elapsed_s

        self._last_fetch_time_s = elapsed_s
        self._throughput_window.append(throughput_mbps)
        return elapsed_s, self._smoothed_throughput()

    def _smoothed_throughput(self) -> float:
        """Sliding-window average throughput (Mbps) over last 5 probes."""
        if not self._throughput_window:
            return 0.0
        return sum(self._throughput_window) / len(self._throughput_window)