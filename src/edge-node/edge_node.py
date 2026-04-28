from __future__ import annotations

import os
import time
import json
import threading
import urllib.request
from collections import deque
from typing import Optional


class HW_Net_API:
    """
    Hardware & Network Sensor API.

    Provides real container hardware metrics (CPU, memory) and
    real network metrics (buffer occupancy, segment fetch time,
    estimated throughput) measured against a target streaming node.

    Usage:
        api = HW_Net_API("metrics.json", state="normal", target_node="cdn.example.com")

        # ABR Engine calls this after each segment download:
        api.update_buffer_occupancy(current_buffer_seconds)

        # TinyRL Model polls every decision step:
        hw  = api.get_hw_state()
        net = api.get_net_state()
    """

    def __init__(self, shared_metrics_path: str, state: str, target_node: str):
        """
        Args:
            shared_metrics_path: Path to JSON file with simulated HW states
                                 (thermal_state, battery_level, etc.)
            state:               Active state key in the metrics JSON
                                 (e.g. "normal", "low_battery", "throttled")
            target_node:         Hostname or IP of the streaming/CDN node
                                 used for network probing.
        """
        with open(shared_metrics_path, "r") as f:
            self.metrics = json.load(f)

        self.state = state
        self.target_node = target_node

        # Buffer occupancy is pushed in by the ABR Engine
        self._buffer_seconds: float = 0.0
        self._buffer_lock = threading.Lock()

        # Sliding window of last 5 throughput samples for smoothing
        self._throughput_window: deque = deque(maxlen=5)

    # =========================================================================
    # Public API
    # =========================================================================

    def get_hw_state(self) -> dict:
        """
        Returns real container CPU/memory alongside simulated
        thermal and battery values from the shared metrics file.

        Returns:
            {
                "cpu_utilization":  float   # % (0-100)
                "memory_pressure":  float   # % (0-100)
                "thermal_state":    str     # e.g. "nominal", "warm", "critical"
                "battery_level":    float   # % (0-100)
            }
        """
        return {
            "cpu_utilization": self._get_cpu_utilization(),
            "memory_pressure": self._get_memory_pressure(),
            "thermal_state":   self.metrics.get(self.state, {}).get("thermal_state"),
            "battery_level":   self.metrics.get(self.state, {}).get("battery_level"),
        }

    def get_net_state(self) -> dict:
        """
        Measures real network conditions from the container to target_node.

        Returns:
            {
                "buffer_occupancy":     float   # seconds
                "segment_fetch_time":   float   # milliseconds
                "estimated_throughput": float   # Mbps
            }
        """
        fetch_time_ms, throughput_bps = self._measure_segment_fetch()
        return {
            "buffer_occupancy":     self._get_buffer_occupancy(),
            "segment_fetch_time":   fetch_time_ms,
            "estimated_throughput": throughput_bps / 1_000_000,  # bps → Mbps
        }

    def update_buffer_occupancy(self, seconds: float) -> None:
        """
        Called by the ABR Engine after each segment download to
        update the current playback buffer level.

        Args:
            seconds: Current buffer size in seconds.
        """
        with self._buffer_lock:
            self._buffer_seconds = max(0.0, seconds)

    def set_state(self, state: str) -> None:
        """
        Switch the active simulated HW state (called by Orchestrator).

        Args:
            state: Key into the shared metrics JSON.
        """
        if state not in self.metrics:
            raise ValueError(f"Unknown state '{state}'. Available: {list(self.metrics.keys())}")
        self.state = state

    # =========================================================================
    # Network Metrics
    # =========================================================================

    def _get_buffer_occupancy(self) -> float:
        with self._buffer_lock:
            return self._buffer_seconds

    def _measure_segment_fetch(self) -> tuple[float, float]:
        """
        Primary: HTTP probe to target_node/probe (lightweight ~100KB endpoint).
        Fallback: Passive throughput estimate from /proc/net/dev.

        Returns:
            (fetch_time_ms, throughput_bps)
        """
        probe_url = f"http://{self.target_node}/probe"
        try:
            return self._http_probe(probe_url)
        except Exception:
            return self._proc_net_fallback()

    def _http_probe(self, url: str) -> tuple[float, float]:
        """
        Download a probe object and measure elapsed time + throughput.
        Updates the sliding window average.
        """
        t_start = time.perf_counter()
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = resp.read()
        t_end = time.perf_counter()

        elapsed_s  = t_end - t_start
        elapsed_ms = elapsed_s * 1000
        size_bits  = len(data) * 8
        throughput_bps = size_bits / elapsed_s if elapsed_s > 0 else 0.0

        self._throughput_window.append(throughput_bps)
        return elapsed_ms, self._smoothed_throughput()

    def _proc_net_fallback(self) -> tuple[float, float]:
        """
        Passive fallback: measure RX bytes on the container's network
        interface over 500ms via /proc/net/dev.

        Returns (0.0 ms fetch time, estimated_bps).
        """
        iface = self._detect_interface()
        if iface is None:
            return 0.0, self._smoothed_throughput()

        def read_rx_bytes() -> int:
            with open("/proc/net/dev") as f:
                for line in f:
                    if iface in line:
                        return int(line.split()[1])  # column 1 = RX bytes
            return 0

        b1 = read_rx_bytes()
        time.sleep(0.5)
        b2 = read_rx_bytes()

        throughput_bps = ((b2 - b1) * 8) / 0.5
        self._throughput_window.append(throughput_bps)
        return 0.0, self._smoothed_throughput()

    def _smoothed_throughput(self) -> float:
        """Return sliding-window average throughput (bps)."""
        if not self._throughput_window:
            return 0.0
        return sum(self._throughput_window) / len(self._throughput_window)

    def _detect_interface(self) -> Optional[str]:
        """Return the first non-loopback interface from /proc/net/dev."""
        try:
            with open("/proc/net/dev") as f:
                for line in f:
                    line = line.strip()
                    if ":" in line:
                        iface = line.split(":")[0].strip()
                        if iface != "lo":
                            return iface
        except Exception:
            pass
        return None

    # =========================================================================
    # Hardware Metrics — CPU
    # =========================================================================

    def _get_cpu_utilization(self) -> float:
        """cgroup v2 → cgroup v1 → psutil fallback chain."""
        try:
            return self._cpu_cgroup_v2()
        except Exception:
            try:
                return self._cpu_cgroup_v1()
            except Exception:
                return self._cpu_psutil()

    def _cpu_cgroup_v2(self) -> float:
        """
        Reads cpu.stat usage_usec, samples over 100ms, returns % utilization.
        """
        cpu_stat = "/sys/fs/cgroup/cpu.stat"
        if not os.path.exists(cpu_stat):
            raise FileNotFoundError(cpu_stat)

        def read_usage_usec() -> int:
            with open(cpu_stat) as f:
                for line in f:
                    if line.startswith("usage_usec"):
                        return int(line.split()[1])
            raise ValueError("usage_usec not found in cpu.stat")

        t1 = read_usage_usec()
        time.sleep(0.1)
        t2 = read_usage_usec()

        delta_us   = t2 - t1
        elapsed_us = 100_000            # 100ms in microseconds
        num_cpus   = os.cpu_count() or 1
        return min((delta_us / elapsed_us / num_cpus) * 100, 100.0)

    def _cpu_cgroup_v1(self) -> float:
        """
        Reads cpuacct.usage (nanoseconds), samples over 100ms.
        """
        usage_file = "/sys/fs/cgroup/cpuacct/cpuacct.usage"
        if not os.path.exists(usage_file):
            raise FileNotFoundError(usage_file)

        def read_ns() -> int:
            with open(usage_file) as f:
                return int(f.read().strip())

        t1 = read_ns()
        time.sleep(0.1)
        t2 = read_ns()

        delta_ns   = t2 - t1
        elapsed_ns = 100_000_000        # 100ms in nanoseconds
        num_cpus   = os.cpu_count() or 1
        return min((delta_ns / elapsed_ns / num_cpus) * 100, 100.0)

    @staticmethod
    def _cpu_psutil() -> float:
        import psutil
        return psutil.cpu_percent(interval=0.1)

    # =========================================================================
    # Hardware Metrics — Memory
    # =========================================================================

    def _get_memory_pressure(self) -> float:
        """cgroup v2 → cgroup v1 → psutil fallback chain."""
        try:
            return self._memory_cgroup_v2()
        except Exception:
            try:
                return self._memory_cgroup_v1()
            except Exception:
                return self._memory_psutil()

    def _memory_cgroup_v2(self) -> float:
        """
        Reads memory.current / memory.max.
        If limit is 'max' (unlimited), falls back to host physical memory.
        """
        usage_file = "/sys/fs/cgroup/memory.current"
        limit_file = "/sys/fs/cgroup/memory.max"
        if not os.path.exists(usage_file):
            raise FileNotFoundError(usage_file)

        with open(usage_file) as f:
            usage = int(f.read().strip())

        with open(limit_file) as f:
            raw = f.read().strip()
            limit = (
                int(raw)
                if raw != "max"
                else os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            )

        return (usage / limit) * 100

    def _memory_cgroup_v1(self) -> float:
        """
        Reads memory.usage_in_bytes / memory.limit_in_bytes.
        A near-max int64 limit means unlimited.
        """
        usage_file = "/sys/fs/cgroup/memory/memory.usage_in_bytes"
        limit_file = "/sys/fs/cgroup/memory/memory.limit_in_bytes"
        if not os.path.exists(usage_file):
            raise FileNotFoundError(usage_file)

        with open(usage_file) as f:
            usage = int(f.read().strip())

        with open(limit_file) as f:
            raw = int(f.read().strip())
            limit = (
                raw
                if raw < 2 ** 62
                else os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            )

        return (usage / limit) * 100

    @staticmethod
    def _memory_psutil() -> float:
        import psutil
        return psutil.virtual_memory().percent

class ABR_RL_Engine:
    def __init__(self, model, action_space, config, hw_net_api):
        """
        model         : tiny NN
        action_space  : list of possible actions
        config        : hyperparameters (gamma, epsilon, lr)
        hw_net_api    : API for accessing hardware and network state (for now baca dari shared file)
        """
        self.model = model
        self.action_space = action_space
        self.hw_net_api = hw_net_api

        self.gamma = config.get("gamma", 0.9)
        self.epsilon = config.get("epsilon", 0.1)
        self.lr = config.get("lr", 0.001)

        self.prev_state = None
        self.prev_action = None

    def encode_state(self, hw_state, net_state):
        """
        Convert raw signals → compact numeric state
        """
        return [
            hw_state["cpu_utilization"],
            hw_state["memory_pressure"],
            hw_state["thermal_state"],
            hw_state["battery_level"],
            net_state["buffer_occupancy"],
            net_state["segment_fetch_time"],
            net_state["throughput"]
        ]

    def decide(self):
        state = self.encode_state(self.hw_net_api.get_hw_state(), self.hw_net_api.get_net_state())

        # forward pass
        q_values = self.model.predict(state)

        # ε-greedy
        if random.random() < self.epsilon:
            action = random.choice(self.action_space)
        else:
            action = self.action_space[int(np.argmax(q_values))]

        # store for learning
        self.prev_state = state
        self.prev_action = action

        return action

    def get_reward(self):
        """
        Get reward based on QoE metrics (e.g. bitrate, rebuffering, smoothness) from cloud-node
        """
        pass

    def update(self, hw_state, net_state, reward):
        if self.prev_state is None:
            return

        next_state = self.encode_state(hw_state, net_state)

        # current Q
        q_values = self.model.predict(self.prev_state)

        # next Q
        next_q_values = self.model.predict(next_state)

        target = q_values.copy()
        a_idx = self.action_space.index(self.prev_action)

        # Q-learning target
        target[a_idx] = reward + self.gamma * max(next_q_values)

        # train model
        self.model.train(self.prev_state, target, lr=self.lr)

if __name__ == "__main__":
    # Example usage
    hw_net_api = HW_Net_API(shared_metrics_path="/app/shared/train_schemes.json", state="optimal_conditions", target_node="cloud_node_container")
    print("HW State:", json.dumps(hw_net_api.get_hw_state(), indent=2))

