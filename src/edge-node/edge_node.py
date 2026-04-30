from __future__ import annotations

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
                                 (thermal_state, battery_level, etc.)
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

    # =========================================================================
    # Public API
    # =========================================================================

    def refresh_state(self):
        """Reload the shared metrics JSON to get updated simulated values."""
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

    def get_hw_state(self) -> dict:
        """
        Returns real container CPU/memory alongside simulated
        thermal and battery values from the shared metrics file.

        Returns:
            {
                "cpu_pressure":  float   # % (0-100)
                "memory_pressure":  float   # % (0-100)
                "thermal_state":    str     # severity scale (0.0-1.0) scaled on 20-100°C
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


def cpu_burn(api: HW_Net_API, duration_s: float = 1.0) -> dict:
    """Dummy workload yang represent inference + compression cycle.
    Return averaged pressure observations during the burn."""
    cpu_samples = []
    mem_samples = []

    wall_t0 = time.time()
    end = wall_t0 + duration_s
    ops = 0
    x = 0.0
    sample_every = 0.2  # sample tiap 200ms
    next_sample = wall_t0 + sample_every

    while time.time() < end:
        # do work
        for _ in range(10_000):
            x += math.sqrt(12345.6789) * math.sin(x)
        ops += 10_000

        # sample observation periodically
        if time.time() >= next_sample:
            cpu_samples.append(api.get_cpu_pressure(sample_window_s=0.05))
            mem_samples.append(api.get_memory_pressure())
            next_sample += sample_every

    return {
        "ops":              ops,
        "wall_elapsed":     time.time() - wall_t0,
        "cpu_avg_pressure": sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0,
        "mem_avg_pressure": sum(mem_samples) / len(mem_samples) if mem_samples else 0.0,
        "cpu_samples":      cpu_samples,
        "mem_samples":      mem_samples,
    }

if __name__ == "__main__":
    api = HW_Net_API(
        shared_status_path="/app/shared/status.json",
        target_node="cloud_node_container",
    )
    print("[edge-node] HW_Net_API ready", flush=True)

    while True:
        burn = cpu_burn(api, duration_s=1.0)
        hw   = api.get_hw_state()
        
        print(
            f"ops={burn['ops']/1e6:.2f}M  "
            f"cpu_avg={burn['cpu_avg_pressure']:5.1f}%  "
            f"mem_avg={burn['mem_avg_pressure']:5.1f}%  "
            f"| thermal={hw['thermal_state']} bat={hw['battery_level']}",
            flush=True,
        )
        time.sleep(2)

