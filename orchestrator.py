import json
import subprocess
import time
import os
from datetime import datetime

SCENARIOS_PATH = "./src/shared/train_hw_schemes.json"
SHARED_PATH    = "./src/shared/status.json"
CONTAINER_NAME = "edge_node_container"
CLOUD_CONTAINER = "cloud_node_container"
BASELINE_S3    = 0.05  # kapasitas total ESP32-S3 di Docker

SCENARIO_DURATION = 150

def f_battery(V_pct: float) -> float:
    """Battery degradation factor. Smooth ramp di bawah 20%."""
    if V_pct >= 20:
        return 1.0
    return 0.5 + 0.025 * V_pct

def f_thermal(T_celsius: float) -> float:
    """Thermal throttling. Mulai di atas 85°C, floor 0.5."""
    if T_celsius <= 85:
        return 1.0
    return max(0.5, 1 - 0.01 * (T_celsius - 85))

def compute_limits(sc: dict):
    f_b = f_battery(sc['battery'])
    f_t = f_thermal(sc['temp'])

    C_limit = BASELINE_S3 * f_b * f_t

def apply_hardware_limit(B_quota_fraction: float):
    """B_quota_fraction = fraction of one core (e.g. 0.038 = 3.8% of 1 core)."""
    pass

def apply_network_conditions(bandwidth_mbps, latency_ms, loss_pct):
    """Apply tc netem qdisc on cloud_node's eth0 egress."""
    # Reset dulu (idempotent — biar bisa re-apply)
    subprocess.run(
        ["docker", "exec", CLOUD_CONTAINER,
         "tc", "qdisc", "del", "dev", "eth0", "root"],
        stderr=subprocess.DEVNULL
    )
    cmd = [
        "docker", "exec", CLOUD_CONTAINER,
        "tc", "qdisc", "add", "dev", "eth0", "root", "netem",
        "rate", f"{bandwidth_mbps}mbit",
        "delay", f"{latency_ms}ms",
        "loss", f"{loss_pct}%"
    ]
    print(f"  → tc: {bandwidth_mbps}Mbps / {latency_ms}ms / {loss_pct}% loss")
    subprocess.run(cmd, check=True)


with open(SCENARIOS_PATH, 'r') as f:
    scenarios = json.load(f)

for i, sc in enumerate(scenarios, 1):
    print(f"\n>>> [{datetime.now():%H:%M:%S}] Scenario {i}/{len(scenarios)}: {sc['name']}")
    print(f"    inputs: T={sc['temp']}°C, V={sc['battery']}%, "
          f"RAM={sc['ram_usage_pct']}%, U_sys={sc['cpu_bg_usage_pct']}%, "
          f"Net={sc['net_load_pct']}%")

    C_limit, B_quota, dbg = compute_limits(sc)
    print(f"    factors: f_batt={dbg['f_battery']:.3f}, f_temp={dbg['f_thermal']:.3f}, "
          f"(1-U_sys)={1-dbg['U_sys']:.3f}, (1-RAM)={1-dbg['ram_ratio']:.3f}")
    print(f"    C_limit={C_limit:.5f}, B_quota={B_quota:.5f}")

    apply_hardware_limit(B_quota)
    apply_network_conditions(sc['bandwidth_mbps'], sc['latency_ms'], sc['loss_pct'])

    # Tulis ke shared volume biar edge-node bisa baca sebagai observation
    os.makedirs(os.path.dirname(SHARED_PATH), exist_ok=True)
    enriched = {**sc, "C_limit": C_limit, "B_quota": B_quota}
    with open(SHARED_PATH, 'w') as f:
        json.dump(enriched, f)

    print(f"    holding for {SCENARIO_DURATION}s...")
    time.sleep(SCENARIO_DURATION)

print("\n>>> All scenarios done.")