from dataclasses import dataclass, field
from typing import Optional

@dataclass
class HWState:
    cpu_pressure:    float  # % (0–100), from cgroup sampling
    memory_pressure: float  # % (0–100), instantaneous
    thermal_state:   float  # raw °C (needs normalization downstream)
    battery_level:   float  # 0.0–1.0

@dataclass
class NetState:
    segment_fetch_time:    float  # seconds, last HTTP probe
    estimated_throughput:  float  # Mbps, smoothed over last 2 probes

@dataclass
class ABRState:
    """Merged snapshot passed into the RL agent."""
    hw:                HWState
    net:               NetState
    last_bitrate_kbps: float        # bitrate chosen for previous segment
    buffer_level_s:    float        # playback buffer in seconds
    segment_index:     int

@dataclass
class BitrateDecision:
    bitrate_level:  int    # index into BITRATE_LEVELS
    bitrate_kbps:   float
    confidence:     float  # action probability from policy head


class ABR_API:
    """
    Bridges HW_Net_API → tinyRL ABR agent → bitrate decision.
    """
    def __init__(self, model_path: str, hw_net_api):
        self.model_path = model_path
        self.hw_net_api = hw_net_api

    def get_next_bitrate(self, segment_index: int, buffer_level_s: float) -> BitrateDecision:
        """
        Main API function to get next bitrate decision.
        
        Args:
            segment_index: Index of the next segment to download.
            buffer_level_s: Current playback buffer level in seconds.
        
        Returns:
            BitrateDecision with chosen bitrate and confidence.
        """
        pass

    def update(self, reward: float, new_buffer_level_s: float):
        """
        Record the observed reward after a segment completes.
        Used for online fine-tuning; no-op if agent is inference-only.

        Args:
            reward:            Scalar QoE signal (e.g. quality − rebuffer_penalty − switch_penalty).
            new_buffer_level_s: Buffer level after the segment was downloaded and pushed.
        """
        pass

    def reset(self):
        """
        Reset session state: clears hidden state, history, and last bitrate.
        Call at the start of each new streaming session.
        """
        pass

    def fetch_hw_state(self) -> HWState:
        """
        Pull hardware metrics from HW_Net_API and wrap in HWState.
        Thin adapter — no transformation yet.
        """
        pass

    def fetch_net_state(self) -> NetState:
        """
        Trigger a network probe via HW_Net_API and wrap result in NetState.
        Note: each call incurs real network I/O; call once per segment.
        """
        pass

    def merge_state(self, hw: HWState, net: NetState, buffer_level_s: float, segment_index: int) -> ABRState:
        """
        Combine HW + network observations with player context into ABRState.

        Args:
            hw:             Latest hardware snapshot.
            net:            Latest network probe result.
            buffer_level_s: Current buffer depth.
            segment_index:  Segment about to be fetched.
        """
        pass

    def preprocess_state(self, state: ABRState) -> list[float]:
        """
        Normalize and flatten ABRState into the fixed-length feature vector
        expected by the tinyRL agent.

        Normalizations applied:
            cpu_pressure      → / 100
            memory_pressure   → / 100
            thermal_state     → (°C − TEMP_MIN) / (TEMP_MAX − TEMP_MIN)
            battery_level     → already 0–1, pass through
            segment_fetch_time→ clip + normalize against expected max RTT
            estimated_throughput → / BW_MAX_MBPS
            last_bitrate_kbps → / max(BITRATE_LEVELS)
            buffer_level_s    → clip + normalize against buffer target

        Returns:
            Flat list[float], length == agent input dimension.
        """
        pass

    def _normalize_thermal(self, temp_celsius: float) -> float:
        """
        Scale raw °C to [0, 1] using _TEMP_MIN_C / _TEMP_MAX_C.
        Clips values outside the defined range.
        """
        pass

    def _load_agent(self, model_path: str) -> None:
        """
        Deserialize tinyRL weights and set model to eval/inference mode.

        Args:
            model_path: Path to the serialized model file.
        """
        pass

    def _run_inference(
        self,
        feature_vector: list[float],
    ) -> tuple[int, float]:
        """
        Single forward pass through the tinyRL policy network.
        Updates self._hidden if the agent is recurrent.

        Args:
            feature_vector: Output of preprocess_state().

        Returns:
            (action_index, confidence)
            action_index maps into BITRATE_LEVELS.
        """
        pass

    def _action_to_decision(self, action_index: int, confidence: float) -> BitrateDecision:
        """
        Convert a raw agent action index into a typed BitrateDecision.

        Args:
            action_index: Integer in [0, len(BITRATE_LEVELS)).
            confidence:   Action probability from the policy head.
        """
        pass