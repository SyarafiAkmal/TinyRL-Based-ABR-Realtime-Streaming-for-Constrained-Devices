"""
abr_api.py — TinyRL ABR for upstream constrained streaming
Pure-NumPy PPO actor-critic. Reward weights adjustable via RewardConfig.

Architecture (OnRL-style):
  State (8 features) → FC(64, tanh) → FC(32, tanh) → Actor head (softmax)
                                                     → Critic head (linear)
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass
from typing import Optional

from hw_net_api import HW_Net_API


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BITRATE_LEVELS: list[float] = [150, 300, 600, 1200, 2400, 4800]   # kbps
_INPUT_DIM        = 8
_HIDDEN1          = 64
_HIDDEN2          = 32
_N_ACTIONS        = len(BITRATE_LEVELS)

_BW_MAX_MBPS      = 100.0
_MAX_RTT_MS       = 2000.0
_MAX_BITRATE_KBPS = max(BITRATE_LEVELS)
_TEMP_MIN_C       = 20.0
_TEMP_MAX_C       = 85.0


# ---------------------------------------------------------------------------
# Reward config
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """
    OnRL-style reward:
      r = alpha   * throughput_util
        - beta    * packet_loss
        - gamma   * rtt_norm
        - delta   * switch_penalty
        - epsilon * hw_stress
        - zeta    * encoding_load * hw_stress

    zeta term: penalizes choosing high bitrate under HW stress.
    encoding_load × hw_stress = product, so it only bites when BOTH
    bitrate is high AND the device is stressed.
    Set zeta=0 to disable (vanilla OnRL).
    """
    alpha:   float = 2.0
    beta:    float = 3.0
    gamma:   float = 0.5
    delta:   float = 1.5
    epsilon: float = 0.1
    zeta:    float = 1.0


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HWState:
    cpu_pressure:    float   # % (0–100)
    memory_pressure: float   # % (0–100)
    thermal_state:   float   # raw °C
    battery_level:   float   # 0.0–1.0

@dataclass
class NetState:
    estimated_throughput:  float   # Mbps
    rtt_ms:                float   # ms
    packet_loss_rate:      float   # 0.0–1.0
    previous_bitrate_kbps: float

@dataclass
class ABRState:
    hw:                HWState
    net:               NetState
    last_bitrate_kbps: float

@dataclass
class BitrateDecision:
    bitrate_level: int
    bitrate_kbps:  float
    confidence:    float


# ---------------------------------------------------------------------------
# Pure-NumPy PPO actor-critic
# ---------------------------------------------------------------------------

class _TinyPPONet:
    def __init__(
        self,
        input_dim:  int   = _INPUT_DIM,
        hidden1:    int   = _HIDDEN1,
        hidden2:    int   = _HIDDEN2,
        n_actions:  int   = _N_ACTIONS,
        lr:         float = 1e-3,
        clip_eps:   float = 0.3,
        gamma:      float = 0.99,
        lam:        float = 0.95,
        epochs:     int   = 8,
        minibatch:  int   = 4,
    ):
        self.lr         = lr
        self.clip_eps   = clip_eps
        self.gamma_disc = gamma
        self.lam        = lam
        self.epochs     = epochs
        self.minibatch  = minibatch

        rng = np.random.default_rng(42)

        def _he(fan_in: int, fan_out: int) -> np.ndarray:
            return (rng.standard_normal((fan_out, fan_in))
                    * np.sqrt(2.0 / fan_in)).astype(np.float32)

        # Shared trunk
        self.W1 = _he(input_dim, hidden1);  self.b1 = np.zeros(hidden1, np.float32)
        self.W2 = _he(hidden1,   hidden2);  self.b2 = np.zeros(hidden2, np.float32)

        # Heads
        self.Wa = _he(hidden2, n_actions);  self.ba = np.zeros(n_actions, np.float32)
        self.Wc = _he(hidden2, 1);          self.bc = np.zeros(1,         np.float32)

        # Rollout buffer
        self._buf_states:    list[np.ndarray] = []
        self._buf_actions:   list[int]        = []
        self._buf_rewards:   list[float]      = []
        self._buf_log_probs: list[float]      = []
        self._buf_values:    list[float]      = []
        self._buf_dones:     list[bool]       = []

    # ------------------------------------------------------------------ #

    @staticmethod
    def _tanh(x): return np.tanh(x)

    @staticmethod
    def _softmax(x):
        e = np.exp(x - x.max())
        return e / e.sum()

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        h1    = self._tanh(self.W1 @ x + self.b1)
        h2    = self._tanh(self.W2 @ h1 + self.b2)
        probs = self._softmax(self.Wa @ h2 + self.ba)
        value = float(self.Wc @ h2 + self.bc)
        return probs, value

    def select_action(self, x: np.ndarray) -> tuple[int, float, float]:
        probs, value = self.forward(x)
        action   = int(np.random.choice(len(probs), p=probs))
        log_prob = float(np.log(probs[action] + 1e-8))
        return action, log_prob, value

    # ------------------------------------------------------------------ #

    def store_transition(
        self, state, action, reward, log_prob, value, done=False
    ):
        self._buf_states.append(state.astype(np.float32))
        self._buf_actions.append(action)
        self._buf_rewards.append(reward)
        self._buf_log_probs.append(log_prob)
        self._buf_values.append(value)
        self._buf_dones.append(done)

    def _compute_gae(self, last_value: float = 0.0):
        rewards = np.array(self._buf_rewards,   np.float32)
        values  = np.array(self._buf_values,    np.float32)
        dones   = np.array(self._buf_dones,     np.float32)
        n       = len(rewards)
        adv     = np.zeros(n, np.float32)
        ret     = np.zeros(n, np.float32)
        gae     = 0.0

        for t in reversed(range(n)):
            nv    = last_value if t == n - 1 else values[t + 1]
            delta = rewards[t] + self.gamma_disc * nv * (1 - dones[t]) - values[t]
            gae   = delta + self.gamma_disc * self.lam * (1 - dones[t]) * gae
            adv[t] = gae
            ret[t] = gae + values[t]

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def update(self, reward: float, done: bool = False) -> None:
        if not self._buf_actions:
            return

        self._buf_rewards[-1] = reward
        self._buf_dones[-1]   = done

        if len(self._buf_rewards) < self.minibatch and not done:
            return

        last_value = 0.0
        if not done:
            _, last_value = self.forward(self._buf_states[-1])

        adv, ret = self._compute_gae(last_value)
        states   = np.array(self._buf_states,    np.float32)
        actions  = np.array(self._buf_actions,   np.int32)
        old_lp   = np.array(self._buf_log_probs, np.float32)

        n = len(states)
        for _ in range(self.epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.minibatch):
                mb = idx[start: start + self.minibatch]
                self._ppo_step(
                    states[mb], actions[mb], old_lp[mb], adv[mb], ret[mb]
                )

        self._buf_states.clear();    self._buf_actions.clear()
        self._buf_rewards.clear();   self._buf_log_probs.clear()
        self._buf_values.clear();    self._buf_dones.clear()

    def _ppo_step(self, states, actions, old_lp, adv, ret):
        batch_size = len(states)
        eps = self.clip_eps

        dW1 = np.zeros_like(self.W1); db1 = np.zeros_like(self.b1)
        dW2 = np.zeros_like(self.W2); db2 = np.zeros_like(self.b2)
        dWa = np.zeros_like(self.Wa); dba = np.zeros_like(self.ba)
        dWc = np.zeros_like(self.Wc); dbc = np.zeros_like(self.bc)

        for i in range(batch_size):
            x  = states[i]; a = actions[i]; A = adv[i]; Rt = ret[i]

            h1     = self._tanh(self.W1 @ x + self.b1)
            h2     = self._tanh(self.W2 @ h1 + self.b2)
            logits = self.Wa @ h2 + self.ba
            probs  = self._softmax(logits)
            value  = float(self.Wc @ h2 + self.bc)

            log_p_new = np.log(probs[a] + 1e-8)
            ratio     = np.exp(log_p_new - old_lp[i])

            d_logits  = np.zeros(_N_ACTIONS, np.float32)
            clip_grad = -A if (1 - eps) <= ratio <= (1 + eps) else 0.0
            d_logits += clip_grad * (-1.0) * (np.eye(_N_ACTIONS)[a] - probs)

            dWa += np.outer(d_logits, h2)
            dba += d_logits

            d_value = 2.0 * (value - Rt)
            dWc    += d_value * h2[np.newaxis, :]
            dbc    += d_value

            d_h2      = self.Wa.T @ d_logits + (self.Wc * d_value).squeeze()
            d_h2_pre  = d_h2 * (1.0 - h2 ** 2)
            dW2 += np.outer(d_h2_pre, h1)
            db2 += d_h2_pre

            d_h1_pre = (self.W2.T @ d_h2_pre) * (1.0 - h1 ** 2)
            dW1 += np.outer(d_h1_pre, x)
            db1 += d_h1_pre

        scale = self.lr / batch_size
        self.W1 -= scale * dW1;  self.b1 -= scale * db1
        self.W2 -= scale * dW2;  self.b2 -= scale * db2
        self.Wa -= scale * dWa;  self.ba -= scale * dba
        self.Wc -= scale * dWc;  self.bc -= scale * dbc

    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        if not path.endswith(".npz"):
            path += ".npz"
        np.savez(path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            Wa=self.Wa, ba=self.ba,
            Wc=self.Wc, bc=self.bc,
        )

    def load(self, path: str) -> None:
        if not path.endswith(".npz"):
            path += ".npz"
        p = np.load(path)
        self.W1 = p["W1"]; self.b1 = p["b1"]
        self.W2 = p["W2"]; self.b2 = p["b2"]
        self.Wa = p["Wa"]; self.ba = p["ba"]
        self.Wc = p["Wc"]; self.bc = p["bc"]


# ---------------------------------------------------------------------------
# Reward calculator
# ---------------------------------------------------------------------------

class RewardCalculator:
    def __init__(self, config: Optional[RewardConfig] = None):
        self.cfg = config or RewardConfig()

    def compute(
        self,
        throughput_mbps:  float,
        packet_loss_rate: float,
        rtt_ms:           float,
        switched_to_gcc:  bool,
        hw:               Optional[HWState] = None,
        chosen_kbps:      float = 0.0,
    ) -> float:
        c = self.cfg

        throughput_util = min(throughput_mbps / _BW_MAX_MBPS, 1.0)
        rtt_norm        = min(rtt_ms / _MAX_RTT_MS, 1.0)
        switch_penalty  = 1.0 if switched_to_gcc else 0.0

        hw_stress = 0.0
        if hw is not None and c.epsilon > 0.0:
            thermal_norm = max(0.0, min(
                (hw.thermal_state - _TEMP_MIN_C) / (_TEMP_MAX_C - _TEMP_MIN_C), 1.0
            ))
            hw_stress = (
                0.4 * (hw.cpu_pressure    / 100.0)
              + 0.3 * (hw.memory_pressure / 100.0)
              + 0.2 * thermal_norm
              + 0.1 * (1.0 - hw.battery_level)
            )

        # Encoding load penalty:
        # Product of bitrate_norm × hw_stress means it only penalises
        # when BOTH the device is stressed AND bitrate is high.
        encoding_load    = chosen_kbps / _MAX_BITRATE_KBPS   # 0.0–1.0
        encoding_penalty = encoding_load * hw_stress

        return float(
            c.alpha   * throughput_util
          - c.beta    * packet_loss_rate
          - c.gamma   * rtt_norm
          - c.delta   * switch_penalty
          - c.epsilon * hw_stress
          - c.zeta    * encoding_penalty
        )


# ---------------------------------------------------------------------------
# ABR_API
# ---------------------------------------------------------------------------

class ABR_API:
    """
    Bridges HW_Net_API → tinyRL PPO agent → upstream bitrate decision.

    Quick-start:
        api = ABR_API("model.npz", hw_net_api)
        decision = api.get_next_bitrate()
        reward = api.reward_calc.compute(tput, loss, rtt, switched,
                                         hw_snap, decision.bitrate_kbps)
        api.update(reward)
    """

    def __init__(
        self,
        model_path:    str,
        hw_net_api:    HW_Net_API,
        reward_config: Optional[RewardConfig] = None,
    ):
        self.model_path        = model_path
        self.hw_net_api        = hw_net_api
        self.last_bitrate_kbps = 0.0
        self.reward_calc       = RewardCalculator(reward_config)
        self._agent            = _TinyPPONet()
        self._load_agent(model_path)

        self._last_log_prob: float               = 0.0
        self._last_value:    float               = 0.0
        self._last_state:    Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #

    def get_next_bitrate(self) -> BitrateDecision:
        abr_state      = self.fetch_abr_state()
        feature_vector = self.preprocess_state(abr_state)

        action_idx, log_prob, value = self._run_inference(feature_vector)
        decision = self._action_to_decision(action_idx, np.exp(log_prob))

        self._last_log_prob    = log_prob
        self._last_value       = value
        self._last_state       = feature_vector.copy()
        self.last_bitrate_kbps = decision.bitrate_kbps
        self.hw_net_api.set_previous_bitrate(decision.bitrate_kbps)

        self._agent.store_transition(
            state    = feature_vector,
            action   = action_idx,
            reward   = 0.0,
            log_prob = log_prob,
            value    = value,
            done     = False,
        )
        return decision

    def update(self, reward: float, done: bool = False) -> None:
        self._agent.update(reward, done=done)

    def reset(self) -> None:
        self._agent._buf_states.clear()
        self._agent._buf_actions.clear()
        self._agent._buf_rewards.clear()
        self._agent._buf_log_probs.clear()
        self._agent._buf_values.clear()
        self._agent._buf_dones.clear()
        self.last_bitrate_kbps = 0.0
        self._last_log_prob    = 0.0
        self._last_value       = 0.0
        self._last_state       = None
        self.hw_net_api.set_previous_bitrate(0.0)

    def save_model(self, path: Optional[str] = None) -> None:
        self._agent.save(path or self.model_path)

    # ------------------------------------------------------------------ #

    def fetch_abr_state(self) -> ABRState:
        hw  = self.hw_net_api.get_hw_state()
        net = self.hw_net_api.get_net_state()
        return ABRState(
            hw=HWState(
                cpu_pressure    = hw["cpu_pressure"],
                memory_pressure = hw["memory_pressure"],
                thermal_state   = hw["thermal_state"],
                battery_level   = hw["battery_level"],
            ),
            net=NetState(
                estimated_throughput  = net["estimated_throughput"],
                rtt_ms                = net["rtt_ms"],
                packet_loss_rate      = net["packet_loss_rate"],
                previous_bitrate_kbps = net["previous_bitrate_kbps"],
            ),
            last_bitrate_kbps = self.last_bitrate_kbps,
        )

    def preprocess_state(self, state: ABRState) -> np.ndarray:
        return np.array([
            state.hw.cpu_pressure    / 100.0,
            state.hw.memory_pressure / 100.0,
            self._normalize_thermal(state.hw.thermal_state),
            state.hw.battery_level,
            min(state.net.estimated_throughput  / _BW_MAX_MBPS, 1.0),
            state.net.packet_loss_rate,
            min(state.net.rtt_ms               / _MAX_RTT_MS, 1.0),
            state.net.previous_bitrate_kbps    / _MAX_BITRATE_KBPS,
        ], dtype=np.float32)

    def _normalize_thermal(self, temp_celsius: float) -> float:
        return float(max(0.0, min(
            (temp_celsius - _TEMP_MIN_C) / (_TEMP_MAX_C - _TEMP_MIN_C), 1.0
        )))

    def _load_agent(self, model_path: str) -> None:
        npz_path = model_path if model_path.endswith(".npz") else model_path + ".npz"
        if os.path.exists(npz_path):
            self._agent.load(npz_path)
            print(f"[abr] loaded model from {npz_path}", flush=True)
        else:
            print(f"[abr] no model found at {npz_path}, starting fresh", flush=True)

    def _run_inference(self, feature_vector: np.ndarray) -> tuple[int, float, float]:
        return self._agent.select_action(feature_vector)

    def _action_to_decision(self, action_index: int, confidence: float) -> BitrateDecision:
        return BitrateDecision(
            bitrate_level = action_index,
            bitrate_kbps  = BITRATE_LEVELS[action_index],
            confidence    = confidence,
        )