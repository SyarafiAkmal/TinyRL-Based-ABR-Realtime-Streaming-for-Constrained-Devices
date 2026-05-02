import json
import subprocess
import time
import os
from datetime import datetime

SCENARIOS_PATH = "./src/shared/train_schemes.json"
SHARED_PATH    = "./src/shared/status.json"
CONTAINER_NAME = "edge_node_container"
CLOUD_CONTAINER = "cloud_node_container"
BASELINE_RP4    = 0.25  # kapasitas total Raspberry Pi 4 di Docker

SCENARIO_DURATION = 20

def f_thermal(T_celsius: float) -> float:
    """RPi4 thermal capacity multiplier.
    1.0 (≤20°C) → 0.4 (≥80°C). Floor 0.4 prevents zero CPU at high temp."""
    severity = max(0.0, min(1.0, (T_celsius - 20) / (80 - 20)))
    return max(0.4, 1.0 - severity)

def compute_c_limit(sc: dict) -> float:
    """C_limit = baseline_RP4 × f_battery × f_thermal."""
    
    f_b = sc['battery_level']
    f_t = f_thermal(sc['temp_celsius'])
    C_limit = BASELINE_RP4 * f_b * f_t

    return C_limit

def apply_hardware_limit(sc: dict):
    """
    Apply cpu quota on edge_node_container.
    
    C_limit = baseline_RP4 × f_battery × f_thermal
    """
    c_limit = compute_c_limit(sc)
    cpus = max(0.01, c_limit)
    
    if c_limit < 0.01:
        print(f"  ⚠ c_limit={c_limit:.5f} below Docker floor 0.01, clamped")

    print(f"  Applying CPU quota: {cpus:.5f} (C_limit={c_limit:.5f})")
    try:
        subprocess.run(
            ["docker", "update", f"--cpus={cpus:.5f}", CONTAINER_NAME],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to update container CPU quota: {e}")


def apply_network_conditions(sc: dict):
    """Apply tc netem qdisc on cloud_node's eth0 egress."""

    rate  = sc.get("rate",  "10mbit")
    delay = sc.get("delay", "0ms")
    loss  = sc.get("loss",  "0%").rstrip("%") 

    subprocess.run(
        ["docker", "exec", CLOUD_CONTAINER,
         "tc", "qdisc", "del", "dev", "eth0", "root"],
        stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
    )

    cmd = [
        "docker", "exec", CLOUD_CONTAINER,
        "tc", "qdisc", "add", "dev", "eth0", "root", "netem",
        "rate",  rate,
        "delay", delay,
        "loss",  loss,
    ]

    print(f"  Applying tc netem: rate={rate} delay={delay} loss={loss}%")

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except:
        print(f"  ❌ Failed to apply network conditions: {cmd}")

with open(SCENARIOS_PATH, 'r') as f:
    scenarios = json.load(f)

for i, sc in enumerate(scenarios):
    # Apply hardware limits based on current state
    print(f"[Scenario {sc}]")
    with open(SHARED_PATH, 'w') as f:
        json.dump(scenarios[sc], f)
    print(f"[Scenario {sc}] Applying hardware limits...")
    apply_hardware_limit(scenarios[sc])

    # Apply network conditions based on current state
    print(f"[Scenario {sc}] Applying network conditions...")
    apply_network_conditions(scenarios[sc])

    time.sleep(SCENARIO_DURATION)
    print(f"[Scenario {sc}] Done.")

print("\n>>> All scenarios done.")