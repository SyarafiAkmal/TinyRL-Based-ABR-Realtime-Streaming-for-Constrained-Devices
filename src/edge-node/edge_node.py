from __future__ import annotations

from hw_net_api import HW_Net_API
import math
import time

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
        target_node="cloud_node_container:8000",  # ← include port
    )
    print("[edge-node] HW_Net_API ready", flush=True)

    while True:
        print("[loop] burn start", flush=True)
        burn = cpu_burn(api, duration_s=1.0)
        print("[loop] burn done, probing net", flush=True)
        hw   = api.get_hw_state()
        net  = api.get_net_state()
        print("[loop] probe done", flush=True)
        print(f"ops={burn['ops']/1e6:.2f}M ...", flush=True)

        print(
            f"ops={burn['ops']/1e6:.2f}M  "
            f"cpu_avg={burn['cpu_avg_pressure']:5.1f}%  "
            f"mem_avg={burn['mem_avg_pressure']:5.1f}%  "
            f"| thermal={hw['thermal_state']} bat={hw['battery_level']}  "
            f"| rtt={net['rtt_ms']:6.1f}ms "
            f"thr={net['estimated_throughput']:5.2f}Mbps "
            f"loss={net['packet_loss_rate']*100:4.1f}% "
            f"prev_br={net['previous_bitrate_kbps']:7.1f}kbps",
            flush=True,
        )
        # time.sleep(2)

